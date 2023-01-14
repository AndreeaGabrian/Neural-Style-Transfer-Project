import os
import subprocess


def make_video_from_images(images_path):
    import shutil
    out_file_name = f'{images_path}.mp4'
    fps = 30
    first_frame = 0
    number_of_frames_to_process = len(os.listdir(images_path))

    ffmpeg = 'ffmpeg'
    if shutil.which(ffmpeg):  # if ffmpeg is in system path
        img_name_format = '%4d.jpg'
        pattern = os.path.join(images_path, img_name_format)
        out_video_path = os.path.join(images_path, out_file_name)

        trim_video_command = ['-start_number', str(first_frame), '-vframes', str(number_of_frames_to_process)]
        input_options = ['-r', str(fps), '-i', pattern]
        encoding_options = ['-c:v', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p']
        subprocess.call(['ffmpeg', *input_options, *trim_video_command, *encoding_options, out_video_path])
    else:
        print(f'{ffmpeg} not found in the system path, aborting.')


make_video_from_images(r"C:\Users\gabri\Documents\AN 3 SEM 1\CVDL\NeuralStyleTransferProject\data\output-images\transferred_bottle_lady2_content_200000.0_3000000.0_1000.0")


# (
#     ffmpeg
#     .input(r'C:\Users\gabri\Documents\AN 3 SEM 1\CVDL\NeuralStyleTransferProject\data\output-images\c_reconstruction_lbfgs/*.jpg', pattern_type='glob', framerate=25)
#     .output('movie.mp4')
#     .run()
# )