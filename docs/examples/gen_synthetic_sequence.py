#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成一份符合 btc_desc 输入规范的合成激光序列（点云帧 + 位姿文件）。

只依赖 numpy，不需要 ROS / PCL。产出的目录可以直接被
``btc_place_recognition`` 读取，用来验证"数据格式对不对"。

用法::

    # 默认：2 圈 x 120 帧 = 240 帧，.bin 格式，输出到 ./btc_demo
    python3 gen_synthetic_sequence.py --out /tmp/btc_demo

    # 少量帧 + PCD 格式（read_bin=false 时使用）
    python3 gen_synthetic_sequence.py --out /tmp/btc_demo_pcd \\
        --laps 1 --frames-per-lap 30 --format pcd

生成结果::

    <out>/velodyne/000000.bin   每点 4 x float32: x y z intensity（传感器系）
    <out>/velodyne/000001.bin   ...
    <out>/pose.txt              每行 8 列: id x y z qx qy qz qw（T_world_lidar）

场景：一个 20 m x 20 m、高 3.5 m 的房间，内部有隔断墙、箱体和立柱；
传感器高 1.6 m，沿正方形闭合路径匀速走 ``--laps`` 圈。走完一圈回到起点时，
第 k 帧与第 k+frames_per_lap 帧观测到同一片结构 —— 这正是回环检测要的东西。
"""

import argparse
import os
import time

import numpy as np


# --------------------------------------------------------------------------
# 场景定义（世界系，z 向上，单位 m）
# --------------------------------------------------------------------------
def make_room(size):
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([size, size, 3.5])
    return lo, hi


def make_boxes(size, seed):
    """返回若干轴对齐长方体 (cx, cy, cz, hx, hy, hz) 作为场景杂波。"""
    boxes = []
    m = size * 0.5
    boxes += [
        (m, size * 0.35, 1.25, size * 0.15, 0.15, 1.25),
        (m, size * 0.65, 1.25, size * 0.15, 0.15, 1.25),
        (size * 0.35, m, 1.25, 0.15, size * 0.15, 1.25),
        (size * 0.65, m, 1.25, 0.15, size * 0.15, 1.25),
    ]
    rng = np.random.RandomState(seed + 1)
    for _ in range(10):
        cx = rng.uniform(size * 0.32, size * 0.68)
        cy = rng.uniform(size * 0.32, size * 0.68)
        hx = rng.uniform(0.4, 1.0)
        hy = rng.uniform(0.4, 1.0)
        hz = rng.uniform(0.4, 1.2)
        boxes.append((cx, cy, hz, hx, hy, hz))
    for cx, cy in [(0.28 * size, 0.28 * size), (0.72 * size, 0.28 * size),
                   (0.28 * size, 0.72 * size), (0.72 * size, 0.72 * size)]:
        boxes.append((cx, cy, 1.75, 0.2, 0.2, 1.75))
    return boxes


def ray_dirs(n_beams, n_az):
    """传感器系射线方向：仰角均匀分布 x 方位角均匀分布。"""
    elev = np.deg2rad(np.linspace(-25.0, 15.0, n_beams))
    az = np.linspace(0.0, 2.0 * np.pi, n_az, endpoint=False)
    el, aa = np.meshgrid(elev, az, indexing="ij")
    el = el.ravel()
    aa = aa.ravel()
    return np.stack([np.cos(el) * np.cos(aa),
                     np.cos(el) * np.sin(aa),
                     np.sin(el)], axis=1)


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------
# 光线投射（向量化）
# --------------------------------------------------------------------------
def _safe_div(num, den):
    """den 接近 0 时返回 inf，避免 0/0 的 RuntimeWarning。"""
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    return num / den


def exit_box(o, d, lo, hi):
    """射线（起点在盒内）到达盒表面的距离：每个轴取正向穿出面的 t，再取最小。"""
    t = []
    for ax in range(3):
        t1 = _safe_div(lo[ax] - o[ax], d[:, ax])
        t2 = _safe_div(hi[ax] - o[ax], d[:, ax])
        t.append(np.maximum(t1, t2))
    return np.min(t, axis=0)


def hit_boxes(o, d, boxes):
    """射线与一组轴对齐盒求交，返回每个方向上最近的命中距离（无命中为 inf）。"""
    t_best = np.full(len(d), np.inf)
    for (cx, cy, cz, hx, hy, hz) in boxes:
        lo = np.array([cx - hx, cy - hy, cz - hz])
        hi = np.array([cx + hx, cy + hy, cz + hz])
        t_enter = np.full(len(d), -np.inf)
        t_exit = np.full(len(d), np.inf)
        for ax in range(3):
            t1 = _safe_div(lo[ax] - o[ax], d[:, ax])
            t2 = _safe_div(hi[ax] - o[ax], d[:, ax])
            t_enter = np.maximum(t_enter, np.minimum(t1, t2))
            t_exit = np.minimum(t_exit, np.maximum(t1, t2))
        hit = (t_exit >= np.maximum(t_enter, 0.0)) & (t_enter > 1e-6)
        t_best = np.where(hit & (t_enter < t_best), t_enter, t_best)
    return t_best


def cast_frame(o, R, dirs_sensor, boxes, room_lo, room_hi, rng,
               noise=0.01, dropout=0.03, max_range=60.0):
    """从位姿 (o, R) 投射一帧，返回传感器系点 (N, 3)。"""
    d_world = dirs_sensor @ R.T
    t = exit_box(o, d_world, room_lo, room_hi)
    t = np.minimum(t, hit_boxes(o, d_world, boxes))
    keep = np.isfinite(t) & (t > 1e-6) & (t < max_range)
    t = t[keep] + rng.normal(0.0, noise, size=int(keep.sum()))
    d_world = d_world[keep]
    p_world = o[None, :] + t[:, None] * d_world
    keep2 = rng.rand(len(t)) > dropout
    p_world = p_world[keep2]
    t = t[keep2]
    # 世界系 -> 传感器系：p_s = R^T (p_w - o)
    p_sensor = (p_world - o[None, :]) @ R
    return p_sensor, t


# --------------------------------------------------------------------------
# 轨迹：正方形闭合路径
# --------------------------------------------------------------------------
def square_path(size, margin, frames_per_lap, laps, height, z_axis=True):
    corners = [(margin, margin), (size - margin, margin),
               (size - margin, size - margin), (margin, size - margin)]
    corners.append(corners[0])
    seglen = [float(np.hypot(corners[i + 1][0] - corners[i][0],
                             corners[i + 1][1] - corners[i][1])) for i in range(4)]
    total = sum(seglen)
    step = total / frames_per_lap
    poses = []
    for _ in range(laps):
        for i in range(frames_per_lap):
            s = i * step
            acc = 0.0
            for j in range(4):
                if s <= acc + seglen[j] or j == 3:
                    u = (s - acc) / seglen[j]
                    x = corners[j][0] + u * (corners[j + 1][0] - corners[j][0])
                    y = corners[j][1] + u * (corners[j + 1][1] - corners[j][1])
                    yaw = np.arctan2(corners[j + 1][1] - corners[j][1],
                                     corners[j + 1][0] - corners[j][0])
                    poses.append((x, y, height, yaw))
                    break
                acc += seglen[j]
    return poses


# --------------------------------------------------------------------------
# 写出：.bin / .pcd / pose.txt
# --------------------------------------------------------------------------
def write_bin(path, xyz, intensity):
    arr = np.concatenate([xyz.astype(np.float32),
                          intensity.astype(np.float32)[:, None]], axis=1)
    arr.astype("<f4").tofile(path)          # x y z intensity，行主序，无头


def write_pcd(path, xyz, intensity):
    arr = np.concatenate([xyz.astype(np.float32),
                          intensity.astype(np.float32)[:, None]], axis=1)
    arr = arr.astype("<f4")
    n = len(arr)
    header = ("# .PCD v0.7 - Point Cloud Data file format\n"
              "VERSION 0.7\n"
              "FIELDS x y z intensity\n"
              "SIZE 4 4 4 4\n"
              "TYPE F F F F\n"
              "COUNT 1 1 1 1\n"
              "WIDTH %d\n"
              "HEIGHT 1\n"
              "VIEWPOINT 0 0 0 1 0 0 0\n"
              "POINTS %d\n"
              "DATA binary\n" % (n, n))
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


def fmt_pose_line(idx, trans, yaw):
    """8 列：id x y z qx qy qz qw（单个空格分隔，不动点 6 位）。"""
    qz = np.sin(yaw * 0.5)
    qw = np.cos(yaw * 0.5)
    return ("%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f"
            % (idx, trans[0], trans[1], trans[2], 0.0, 0.0, qz, qw))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="生成 btc_desc 格式的合成点云序列",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", default="./btc_demo", help="输出目录")
    ap.add_argument("--laps", type=int, default=2, help="绕圈数（>=2 才会产生回环）")
    ap.add_argument("--frames-per-lap", type=int, default=120, help="每圈帧数")
    ap.add_argument("--format", choices=["bin", "pcd"], default="bin",
                    help="点云格式；bin 对应 read_bin=true，pcd 对应 false")
    ap.add_argument("--beams", type=int, default=32, help="线数（仰角 -25°~15° 均匀分布）")
    ap.add_argument("--az", type=int, default=720, help="方位角采样数")
    ap.add_argument("--room-size", type=float, default=20.0, help="房间边长 (m)")
    ap.add_argument("--margin", type=float, default=4.0, help="路径相对房间边界的余量 (m)")
    ap.add_argument("--height", type=float, default=1.6, help="传感器高度 (m)")
    ap.add_argument("--noise", type=float, default=0.01, help="测距噪声标准差 (m)")
    ap.add_argument("--dropout", type=float, default=0.03, help="丢点率 0~1")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    args = ap.parse_args()

    if args.laps * args.frames_per_lap <= 0:
        ap.error("--laps 与 --frames-per-lap 必须为正")
    out = os.path.abspath(args.out)
    cloud_dir = os.path.join(out, "velodyne")
    os.makedirs(cloud_dir, exist_ok=True)

    room_lo, room_hi = make_room(args.room_size)
    boxes = make_boxes(args.room_size, args.seed)
    dirs = ray_dirs(args.beams, args.az)
    poses = square_path(args.room_size, args.margin, args.frames_per_lap,
                        args.laps, args.height)
    rng = np.random.RandomState(args.seed)

    t0 = time.time()
    lines = []
    counts = []
    ext = ".bin" if args.format == "bin" else ".pcd"
    writer = write_bin if args.format == "bin" else write_pcd
    for i, (x, y, z, yaw) in enumerate(poses):
        R = rot_z(yaw)
        o = np.array([x, y, z])
        p_sensor, dist = cast_frame(o, R, dirs, boxes, room_lo, room_hi, rng,
                                    noise=args.noise, dropout=args.dropout)
        # intensity 只为凑满 .bin 的 4 个 float32；btc_desc 全程不读它
        intensity = np.clip(200.0 * np.exp(-dist / 25.0) + rng.uniform(0, 30, len(dist)),
                            0, 255)
        writer(os.path.join(cloud_dir, "%06d%s" % (i, ext)), p_sensor, intensity)
        lines.append(fmt_pose_line(i, o, yaw))
        counts.append(len(p_sensor))
        if i % 40 == 0 or i == len(poses) - 1:
            print("frame %3d/%d: %6d pts" % (i, len(poses) - 1, len(p_sensor)))

    pose_path = os.path.join(out, "pose.txt")
    with open(pose_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    counts = np.asarray(counts)
    total_mb = sum(os.path.getsize(os.path.join(cloud_dir, f))
                   for f in os.listdir(cloud_dir)) / 1024.0 / 1024.0
    print("")
    print("out          : %s" % out)
    print("frames       : %d  (点云 %d 个 %s 文件 + 1 个 pose.txt)" %
          (len(poses), len(poses), ext))
    print("points/frame : min %d  mean %.0f  max %d" %
          (counts.min(), counts.mean(), counts.max()))
    print("disk         : %.1f MB   (generated in %.1fs)" % (total_mb, time.time() - t0))
    print("")
    print("跑它（另开终端先 roscore）：")
    print("  source ~/project/btc_descriptor/devel/setup.bash")
    print("  rosparam set /pcds_dir  %s" % cloud_dir)
    print("  rosparam set /pose_file %s" % pose_path)
    print("  rosparam set /setting_path $(rospack find btc_desc)/config/config_indoor.yaml")
    print("  rosparam set /read_bin %s" % ("true" if args.format == "bin" else "false"))
    print("  rosrun btc_desc btc_place_recognition")


if __name__ == "__main__":
    main()
