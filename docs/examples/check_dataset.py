#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查一份数据的格式是否符合 btc_desc 输入规范（生成的最后一步自检）。

只依赖 numpy。它不跑算法，只做"读得进去吗 / 读进去语义对吗"这类静态检查，
覆盖 btc_desc 会因为数据不规范而**静默出错**的全部要点：

* 帧文件是否从 000000 连续编号、数量是否与 pose 行数一致
* ``.bin`` 是否是严格的 4 x float32（16 字节/点），有没有 NaN/Inf
* ``.pcd`` 头字段是否合法（仅 ascii / binary 两种 DATA 模式）
* pose 行是否恰好 8 个 token、是否单空格分隔、四元数是否单位长度
* 位姿轨迹是否连续（相邻帧位移异常大通常意味着单位/坐标系搞混了）

用法::

    python3 check_dataset.py --data /tmp/btc_demo
    python3 check_dataset.py --pcds-dir /data/kitti/velodyne --pose-file /data/kitti/pose.txt
    python3 check_dataset.py --data /tmp/btc_demo_pcd --format pcd --sample 50

退出码：0 全通过；1 有 ERROR（数据一定读不对）；有 WARN 仍返回 0。
"""

import argparse

import os

import sys

import numpy as np

MAX_SHOW = 5


class Report(object):
    def __init__(self):
        self.errors = []
        self.warns = []
        self.oks = []

    def err(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warns.append(msg)

    def ok(self, msg):
        self.oks.append(msg)

    def dump(self):
        for m in self.oks:
            print("  [ OK ] %s" % m)
        shown = 0
        for m in self.warns:
            if shown < MAX_SHOW:
                print("  [WARN] %s" % m)
            shown += 1
        if shown > MAX_SHOW:
            print("  [WARN] ... 其余 %d 条同类警告省略" % (shown - MAX_SHOW))
        shown = 0
        for m in self.errors:
            if shown < MAX_SHOW:
                print("  [FAIL] %s" % m)
            shown += 1
        if shown > MAX_SHOW:
            print("  [FAIL] ... 其余 %d 条同类错误省略" % (shown - MAX_SHOW))


# --------------------------------------------------------------------------
def check_pose_file(path, rep, max_jump):
    if not os.path.isfile(path):
        rep.err("位姿文件不存在: %s" % path)
        return []
    raw = open(path, "rb").read().decode("utf-8", "replace").splitlines()
    rows = []
    bad_cols = 0
    bad_space = 0
    for lineno, line in enumerate(raw, 1):
        if not line.strip():
            rep.warn("pose.txt:%d 是空行（被忽略，不产生位姿）" % lineno)
            continue
        toks = line.split(" ")
        if len(toks) != 8:
            bad_cols += 1
            if bad_cols <= MAX_SHOW:
                rep.err("pose.txt:%d 有 %d 个 token（必须恰好 8 个）: %r"
                        % (lineno, len(toks), line[:80]))
            continue
        if "" in toks:
            bad_space += 1
            if bad_space <= MAX_SHOW:
                rep.err("pose.txt:%d 含连续空格/行首空格，会被解析成空 token: %r"
                        % (lineno, line[:80]))
            continue
        try:
            v = [float(t) for t in toks]
        except ValueError:
            rep.err("pose.txt:%d 含非数字 token: %r" % (lineno, line[:80]))
            continue
        rows.append(v)
    if not rows:
        rep.err("位姿文件没有解析出任何位姿")
        return []
    rep.ok("位姿行数 %d（每行 8 列，单空格分隔）" % len(rows))

    arr = np.asarray(rows, dtype=np.float64)
    q = arr[:, 4:8]
    qn = np.linalg.norm(q, axis=1)
    badq = np.where(np.abs(qn - 1.0) > 1e-3)[0]
    if len(badq):
        rep.err("%d 行四元数不是单位长度，例如第 %d 行 ||q||=%.6f；"
                "btc_desc 不做归一化，会得到缩放后的旋转矩阵"
                % (len(badq), badq[0] + 1, qn[badq[0]]))
    else:
        rep.ok("四元数均为单位长度（max |‖q‖-1| = %.2e）" % np.abs(qn - 1.0).max())

    if np.allclose(arr[:, 3], arr[0, 3]) and abs(arr[0, 3]) > 5.0:
        rep.warn("z 恒为 %.3f 且偏大，确认高度单位是米（不是毫米）" % arr[0, 3])

    d = np.linalg.norm(np.diff(arr[:, 1:4], axis=0), axis=1)
    if len(d):
        i = int(np.argmax(d))
        if d.max() > max_jump:
            rep.warn("最大帧间位移 %.2f m（第 %d->%d 行），超过 %.1f m；"
                     "若轨迹本身是连续的，多半是坐标系/单位写错了"
                     % (d.max(), i + 1, i + 2, max_jump))
        else:
            rep.ok("轨迹连续：相邻帧位移 max %.2f m，mean %.2f m"
                   % (d.max(), d.mean()))
        di = np.arange(1, len(arr))[d < 1e-9]
        if len(di) > 0.05 * len(arr):
            rep.warn("%d 对相邻位姿完全相同（疑似重复帧）" % len(di))
    return arr


# --------------------------------------------------------------------------
def read_pcd_points(path, rep):
    """解析 PCD 头 + 点数据，返回 (N x stride 的 float64 数组, mode, fields)。"""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        rep.err("无法打开 %s: %s" % (path, e))
        return None
    idx = data.find(b"\nDATA ")
    if idx < 0:
        rep.err("%s 找不到 DATA 行（PCD 头不完整）" % path)
        return None
    end = data.find(b"\n", idx + 1)
    header = data[: end + 1].decode("ascii", "replace")
    body = data[end + 1:]
    n = None
    mode = None
    fields = []
    for line in header.splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0].upper()
        if key == "POINTS" and len(parts) > 1:
            n = int(parts[1])
        elif key == "DATA" and len(parts) > 1:
            mode = parts[1].lower()
        elif key == "FIELDS":
            fields = parts[1:]
    if n is None:
        rep.err("%s 头部缺 POINTS" % path)
        return None
    if mode not in ("ascii", "binary"):
        rep.err("%s 的 DATA=%s；OpenCV PCDReader 只支持 ascii / binary"
                "（binary_compressed 会读取失败）" % (path, mode))
        return None
    if not {"x", "y", "z"}.issubset(set(fields)):
        rep.err("%s 的 FIELDS=%s，必须包含 x y z" % (path, fields))
        return None
    stride = len(fields)
    if mode == "ascii":
        vals = np.fromstring(body.decode("ascii", "replace"), sep=" ")
        if vals.size < n * stride:
            rep.err("%s 头写 POINTS=%d，实际只读出 %d 个值"
                    % (path, n, vals.size))
            return None
        pts = vals[: n * stride].reshape(n, stride)
    else:
        need = n * 4 * stride
        if len(body) < need:
            rep.err("%s 头部 POINTS=%d 需要 %d 字节，实际只有 %d 字节"
                    % (path, n, need, len(body)))
            return None
        pts = np.frombuffer(body[:need], dtype="<f4").reshape(n, stride)
    return pts.astype(np.float64), mode, fields


def load_bin_points(path, rep):
    size = os.path.getsize(path)
    if size == 0:
        rep.err("%s 是空文件" % path)
        return None
    if size % 16 != 0:
        rep.err("%s 大小 %d 不是 16 的整数倍；.bin 必须是每点 4 x float32"
                % (path, size))
        return None
    arr = np.fromfile(path, dtype="<f4")
    pts = arr.reshape(-1, 4)
    return pts


def check_clouds(cloud_dir, fmt, n_expect, rep, sample):
    if not os.path.isdir(cloud_dir):
        rep.err("点云目录不存在: %s" % cloud_dir)
        return None
    ext = "." + fmt
    files = sorted(f for f in os.listdir(cloud_dir) if f.endswith(ext))
    idx = []
    for f in files:
        stem = f[:-len(ext)]
        if not (len(stem) == 6 and stem.isdigit()):
            rep.err("文件名 %s 不符合 6 位零填充序号（%%06d%s）" % (f, ext))
            continue
        idx.append(int(stem))
    if not idx:
        rep.err("目录里没有任何 %s 文件" % ext)
        return None
    idx = np.asarray(idx)
    missing = sorted(set(range(idx.max() + 1)) - set(idx.tolist()))
    if missing:
        rep.err("序号不连续：缺少 %d 个（首个缺失 %06d）；"
                "btc_desc 读不到文件就 break，后面的帧全部不会被处理"
                % (len(missing), missing[0]))
    else:
        rep.ok("帧文件序号 000000 ~ %06d 连续，共 %d 帧" % (idx.max(), len(idx)))
    if n_expect and len(idx) != n_expect:
        msg = ("帧文件数 %d != 位姿行数 %d" % (len(idx), n_expect))
        if len(idx) < n_expect:
            rep.err(msg + "；缺的帧会让主循环静默中止")
        else:
            rep.warn(msg + "；多出来的帧不会被读取")

    order = idx[np.argsort(idx)]
    pick = list(order[:: max(1, len(order) // max(1, sample))])[:sample]
    if order[-1] not in pick:
        pick.append(int(order[-1]))
    npick = 0
    nn = 0
    bbox_lo = np.full(3, np.inf)
    bbox_hi = np.full(3, -np.inf)
    for i in pick:
        path = os.path.join(cloud_dir, "%06d%s" % (i, ext))
        if fmt == "bin":
            pts = load_bin_points(path, rep)
            if pts is None:
                continue
            xyz = pts[:, :3].astype(np.float64)
            if not np.isfinite(pts).all():
                rep.err("%s 含 NaN/Inf" % path)
                continue
        else:
            got = read_pcd_points(path, rep)
            if got is None:
                continue
            pts, _, _ = got
            xyz = pts[:, :3]
            if not np.isfinite(xyz).all():
                rep.err("%s 含 NaN/Inf" % path)
                continue
        npick += 1
        nn += len(xyz)
        bbox_lo = np.minimum(bbox_lo, xyz.min(axis=0))
        bbox_hi = np.maximum(bbox_hi, xyz.max(axis=0))
        r = np.linalg.norm(xyz, axis=1)
        if len(xyz) < 1000:
            rep.warn("%06d 只有 %d 点，太稀疏（室内建议 >=1e4，室外 >=4e4）"
                     % (i, len(xyz)))
        if r.max() > 200.0:
            rep.warn("%06d 最远点 %.0f m，确认单位是米" % (i, r.max()))

    if npick:
        rep.ok("抽检 %d 帧：平均 %.0f 点/帧，传感器系包围盒 "
               "x[%.1f,%.1f] y[%.1f,%.1f] z[%.1f,%.1f]"
               % (npick, nn / float(npick),
                  bbox_lo[0], bbox_hi[0], bbox_lo[1], bbox_hi[1],
                  bbox_lo[2], bbox_hi[2]))
    return npick


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="检查 btc_desc 输入数据格式",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", help="包含 velodyne/ 与 pose.txt 的目录")
    ap.add_argument("--pcds-dir", help="点云帧目录（与 --data 二选一）")
    ap.add_argument("--pose-file", help="位姿文件")
    ap.add_argument("--format", choices=["bin", "pcd"], default=None,
                    help="默认按目录里的文件后缀自动判断（优先 .bin）")
    ap.add_argument("--sample", type=int, default=20, help="抽检多少帧")
    ap.add_argument("--max-jump", type=float, default=5.0,
                    help="相邻帧位移告警阈值 (m)")
    args = ap.parse_args()

    cloud_dir = args.pcds_dir
    pose_file = args.pose_file
    if args.data:
        d = os.path.abspath(args.data)
        if cloud_dir is None:
            sub = os.path.join(d, "velodyne")
            cloud_dir = sub if os.path.isdir(sub) else d
        pose_file = pose_file or os.path.join(d, "pose.txt")
    if not cloud_dir or not pose_file:
        ap.error("需要 --data，或同时给 --pcds-dir 与 --pose-file")

    fmt = args.format
    if fmt is None:
        fmt = "bin" if os.path.isfile(os.path.join(cloud_dir, "000000.bin")) else "pcd"

    print("检查目录 : %s" % cloud_dir)
    print("位姿文件 : %s" % pose_file)
    print("点云格式 : %s" % fmt)
    print("")

    rep = Report()
    poses = check_pose_file(pose_file, rep, args.max_jump)
    print("[1] 位姿文件")
    rep.dump()
    rep2 = Report()
    check_clouds(cloud_dir, fmt, (len(poses) if poses is not None else None),
                 rep2, args.sample)
    print("[2] 点云帧")
    rep2.dump()

    n_err = len(rep.errors) + len(rep2.errors)
    n_warn = len(rep.warns) + len(rep2.warns)
    print("")
    if n_err:
        print("结论：发现 %d 个 ERROR —— 这份数据 btc_desc 一定读不对，先修。" % n_err)
        return 1
    print("结论：格式合规%s。" % ("" if not n_warn else "，但有 %d 条 WARN 建议看一眼" % n_warn))
    print("      下一步：rosrun 跑一遍，确认日志里出现 [Description] reference plane")
    return 0


if __name__ == "__main__":
    sys.exit(main())
