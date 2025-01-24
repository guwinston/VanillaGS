import numpy as np

def decode_int64(value):
    value = np.array(value, dtype=np.int64)
    low_32 = value & np.int64(0xFFFFFFFF)
    high_32 = (value >> np.int64(32)) & np.int64(0xFFFFFFFF)
    return high_32, low_32


N = 2
cam_id = 1
conic_opacity = np.fromfile(r"D:\code\VanillaGS\output\debug\conic_opacity.bin", dtype=np.float32).reshape(-1, 4)
P = conic_opacity.shape[0]
conic_opacity_batch0 = np.fromfile(r"D:\code\VanillaGS\output\debug\conic_opacity_batch.bin", dtype=np.float32).reshape(N, -1, 4)
conic_opacity_batch = np.fromfile(r"D:\code\VanillaGS\output\debug\conic_opacity_batch.bin", dtype=np.float32).reshape(N, -1, 4)[cam_id, ...]
print("conic_opacity", np.allclose(conic_opacity, conic_opacity_batch), np.sum(np.abs(conic_opacity - conic_opacity_batch)))

depths = np.fromfile(r"D:\code\VanillaGS\output\debug\depths.bin", dtype=np.float32)
depths_batch = np.fromfile(r"D:\code\VanillaGS\output\debug\depths_batch.bin", dtype=np.float32).reshape(N, depths.shape[0])[cam_id, ...]
print("depth", np.allclose(depths, depths_batch), np.sum(np.abs(depths - depths_batch)))

means2d = np.fromfile(r"D:\code\VanillaGS\output\debug\means2d.bin", dtype=np.float32).reshape(-1,2)
means2d_batch = np.fromfile(r"D:\code\VanillaGS\output\debug\means2d_batch.bin", dtype=np.float32).reshape(N, -1, 2)[cam_id, ...]
print("means2d", np.allclose(means2d, means2d_batch))

offsets = np.fromfile(r"D:\code\VanillaGS\output\debug\d_offsets_batch.bin", dtype=np.int32)
print(offsets)
point_list = np.fromfile(r"D:\code\VanillaGS\output\debug\point_list.bin", dtype=np.uint32)
S1 = offsets[cam_id]
E1 = offsets[cam_id+1]
point_list_batch = np.fromfile(r"D:\code\VanillaGS\output\debug\point_list_batch.bin", dtype=np.uint32)[S1:E1, ...]
print("point_list", np.allclose(point_list, point_list_batch))

point_list_keys = np.fromfile(r"D:\code\VanillaGS\output\debug\point_list_keys.bin", dtype=np.uint64)
point_list_keys_batch = np.fromfile(r"D:\code\VanillaGS\output\debug\point_list_keys_batch.bin", dtype=np.uint64)[S1:E1, ...]
print("point_list_keys", np.allclose(point_list_keys, point_list_keys_batch))
for i in range(point_list_keys.shape[0]):
    if not np.allclose(point_list_keys[i], point_list_keys_batch[i]):
        high_32, low_32 = decode_int64(point_list_keys[i])
        high_32_batch, low_32_batch = decode_int64(point_list_keys_batch[i])
        print(i, point_list_keys[i], point_list_keys_batch[i], low_32, low_32_batch, high_32, high_32_batch)
        break

point_list_unsorted = np.fromfile(r"D:\code\VanillaGS\output\debug\point_list_unsorted.bin", dtype=np.uint32)
point_list_unsorted_batch = np.fromfile(r"D:\code\VanillaGS\output\debug\point_list_unsorted_batch.bin", dtype=np.uint32)[S1:E1, ...]
print("point_list_unsorted", np.allclose(point_list_unsorted, point_list_unsorted_batch))

point_list_keys_unsorted = np.fromfile(r"D:\code\VanillaGS\output\debug\point_list_keys_unsorted.bin", dtype=np.uint64)
point_list_keys_unsorted_batch = np.fromfile(r"D:\code\VanillaGS\output\debug\point_list_keys_unsorted_batch.bin", dtype=np.uint64)[S1:E1, ...]
print("point_list_key_unsorted", np.allclose(point_list_keys_unsorted, point_list_keys_unsorted_batch))

radii = np.fromfile(r"D:\code\VanillaGS\output\debug\radii.bin", dtype=np.int32)
radii_batch = np.fromfile(r"D:\code\VanillaGS\output\debug\radii_batch.bin", dtype=np.int32).reshape(N,-1)[cam_id, ...]
print("radii", np.allclose(radii, radii_batch), np.sum(np.abs(radii - radii_batch)))

height = 3361
width = 5187
num_tiles = ((height + 15) // 16) * ((width + 15) // 16)
ranges = np.fromfile(r"D:\code\VanillaGS\output\debug\ranges.bin", dtype=np.uint32).reshape(-1,2)[:num_tiles]
tiles = ranges.shape[0]
ranges_batch = np.fromfile(r"D:\code\VanillaGS\output\debug\ranges_batch.bin", dtype=np.uint32).reshape(N,-1,2)[cam_id, ...] - offsets[cam_id]
print("ranges", np.allclose(ranges, ranges_batch), np.sum(np.abs(ranges - ranges_batch)))
for i in range(tiles):
    if not np.allclose(ranges[i], ranges_batch[i]):
        print(i, ranges[i], ranges_batch[i])
        break

rgb = np.fromfile(r"D:\code\VanillaGS\output\debug\rgb.bin", dtype=np.float32).reshape(-1,3)
rgb_batch = np.fromfile(r"D:\code\VanillaGS\output\debug\rgb_batch.bin", dtype=np.float32).reshape(N,-1,3)[cam_id, ...]
print("rgb", np.allclose(rgb, rgb_batch), np.sum(np.abs(rgb - rgb_batch)))
# for i in range(P):
#     if not np.allclose(rgb[i], rgb_batch[i]):
#         print(i, rgb[i], rgb_batch[i])
#         break

tiles_touched = np.fromfile(r"D:\code\VanillaGS\output\debug\tiles_touched.bin", dtype=np.uint32)
tiles_touched_batch = np.fromfile(r"D:\code\VanillaGS\output\debug\tiles_touched_batch.bin", dtype=np.uint32).reshape(N,-1)[cam_id, ...]
print("tiles_touched", np.allclose(tiles_touched, tiles_touched_batch))




