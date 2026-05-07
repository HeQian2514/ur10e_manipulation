import pyrealsense2 as rs
import numpy as np
import cv2

def rl_input():
    ctx = rs.context()

    # Store mouse position and depth value
    mouse_info = {'x': 0, 'y': 0, 'depth': 0}
    
    # Mouse callback function to track position
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_info['x'] = x
            mouse_info['y'] = y
    
    # Create window and set mouse callback
    cv2.namedWindow('aligned depth image')
    cv2.setMouseCallback('aligned depth image', mouse_callback)

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config object to configure the pipeline 

    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start the pipeline with the configuration
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color) # create align object for depth-color alignment
    
    # Get depth scale factor to convert depth values to meters
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # ===== 获取并打印相机内参 =====
    # 获取深度流和彩色流的内参
    depth_stream = profile.get_stream(rs.stream.depth)
    color_stream = profile.get_stream(rs.stream.color)
    
    depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    print("\n" + "="*60)
    print("RealSense D435i 相机内参信息")
    print("="*60)
    
    print("\n【深度相机内参】")
    print(f"分辨率: {depth_intrinsics.width} x {depth_intrinsics.height}")
    print(f"焦距 (fx, fy): ({depth_intrinsics.fx:.4f}, {depth_intrinsics.fy:.4f})")
    print(f"光心 (ppx, ppy): ({depth_intrinsics.ppx:.4f}, {depth_intrinsics.ppy:.4f})")
    print(f"畸变模型: {depth_intrinsics.model}")
    print(f"畸变系数: {depth_intrinsics.coeffs}")
    
    print("\n【彩色相机内参】")
    print(f"分辨率: {color_intrinsics.width} x {color_intrinsics.height}")
    print(f"焦距 (fx, fy): ({color_intrinsics.fx:.4f}, {color_intrinsics.fy:.4f})")
    print(f"光心 (ppx, ppy): ({color_intrinsics.ppx:.4f}, {color_intrinsics.ppy:.4f})")
    print(f"畸变模型: {color_intrinsics.model}")
    print(f"畸变系数: {color_intrinsics.coeffs}")
    
    print("\n【其他信息】")
    print(f"深度缩放因子: {depth_scale:.6f} (用于将深度值转换为米)")
    print("="*60 + "\n")
    
    # 可选：以矩阵形式打印内参
    print("【相机内参矩阵格式】")
    print("\n深度相机内参矩阵 K_depth:")
    print(f"[{depth_intrinsics.fx:10.4f}  {0:10.4f}  {depth_intrinsics.ppx:10.4f}]")
    print(f"[{0:10.4f}  {depth_intrinsics.fy:10.4f}  {depth_intrinsics.ppy:10.4f}]")
    print(f"[{0:10.4f}  {0:10.4f}  {1:10.4f}]")
    
    print("\n彩色相机内参矩阵 K_color:")
    print(f"[{color_intrinsics.fx:10.4f}  {0:10.4f}  {color_intrinsics.ppx:10.4f}]")
    print(f"[{0:10.4f}  {color_intrinsics.fy:10.4f}  {color_intrinsics.ppy:10.4f}]")
    print(f"[{0:10.4f}  {0:10.4f}  {1:10.4f}]")
    
    # 按照 d2p 函数的 camera_intrinsics 数组格式打印
    print("\n【用于 d2p 函数的内参数组格式】")
    print("深度相机内参数组（用于点云转换）:")
    print("camera_intrinsics = np.array([fx, fy, cx, cy, height, width, depth_scale])")
    depth_intrinsics_array = [
        depth_intrinsics.fx,
        depth_intrinsics.fy,
        depth_intrinsics.ppx,
        depth_intrinsics.ppy,
        depth_intrinsics.height,
        depth_intrinsics.width,
        depth_scale
    ]
    print(f"camera_intrinsics = np.array([", end="")
    for i, val in enumerate(depth_intrinsics_array):
        if i < len(depth_intrinsics_array) - 1:
            print(f"{val:.8e}, ", end="")
        else:
            print(f"{val:.8e}])")
    
    print("\n彩色相机内参数组（用于点云转换）:")
    print("camera_intrinsics = np.array([fx, fy, cx, cy, height, width, depth_scale])")
    color_intrinsics_array = [
        color_intrinsics.fx,
        color_intrinsics.fy,
        color_intrinsics.ppx,
        color_intrinsics.ppy,
        color_intrinsics.height,
        color_intrinsics.width,
        depth_scale
    ]
    print(f"camera_intrinsics = np.array([", end="")
    for i, val in enumerate(color_intrinsics_array):
        if i < len(color_intrinsics_array) - 1:
            print(f"{val:.8e}, ", end="")
        else:
            print(f"{val:.8e}])")
    
    print("="*60 + "\n")
    # ===== 内参打印结束 =====

    try:
        while True:
            # wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            if not aligned_frames:
                continue # if no frames, continue to next iteration

            color_frame = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not aligned_depth_frame:
                continue

            # Convert aligned frame to numpy array
            aligned_depth_array = np.asanyarray(aligned_depth_frame.get_data())
            color_frame = np.asanyarray(color_frame.get_data())

            # Get depth value at mouse position
            if (0 <= mouse_info['x'] < aligned_depth_array.shape[1] and 
                0 <= mouse_info['y'] < aligned_depth_array.shape[0]):
                depth_value = aligned_depth_array[mouse_info['y'], mouse_info['x']]
                depth_meters = depth_value * depth_scale
                mouse_info['depth'] = depth_meters

            # Create visualization of depth image
            aligned_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_array, alpha=0.05), 
                                                      cv2.COLORMAP_JET)
            
            # Create visualization for depth display (for showing depth information)
            # Convert depth image to 3-channel color image to add colored text
            depth_display = cv2.cvtColor(cv2.convertScaleAbs(aligned_depth_array, alpha=0.05), cv2.COLOR_GRAY2BGR)
            
            # Mark mouse position on image
            cv2.drawMarker(depth_display, (mouse_info['x'], mouse_info['y']), 
                          (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
            
            # Add depth information text in bottom-left corner
            depth_text = f"Depth: {mouse_info['depth']:.3f} m ({aligned_depth_array[mouse_info['y'], mouse_info['x']]} raw)"
            cv2.putText(depth_display, depth_text, (10, depth_display.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # display the aligned depth frames
            cv2.imshow('Aligned Depth colormap', aligned_depth_colormap)
            cv2.imshow('aligned depth image', depth_display)
            cv2.imwrite('/home/hq/PROJECT/FlowPolicy/real_world/check_pkg/image/aligned_depth_image.png', aligned_depth_array)

            # display the color frame
            cv2.imshow('Color image', color_frame)
            cv2.imwrite('/home/hq/PROJECT/FlowPolicy/real_world/check_pkg/image/color_image.png', color_frame)

            # y_slice = slice(130, 670)  # Y range
            # x_slice = slice(660, 1160)  # X range
            # y_slice = slice(130, 670)  # Y range
            # x_slice = slice(600, 1250)  # X range
            y_slice = slice(10, 710)  # Y range
            x_slice = slice(460, 1200)  # X range

            cropped_rgb = color_frame[y_slice, x_slice]
            cv2.imshow('Cropped RGB', cropped_rgb)

            # press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the pipeline
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    rl_input()