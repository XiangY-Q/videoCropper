# VideoCropper
# Xiangyu Qu, Oct 2018

import argparse
import cv2
import numpy as np
import os.path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VideoCropper is a mini program for helping computer vision '
                                                 'researchers to conveniently hand-crop a large amount of image '
                                                 'patches from a video sequence and save these image patches using '
                                                 'naming rule given by VideoCropper user')
    parser.add_argument("file_path", help='path to video file')
    parser.add_argument("-k", "--num_class", type=int, default=1, help='number of classes of patches')
    parser.add_argument("-p", "--class_prefix", nargs='+', help='prefix of names for different classes of image patches')
    parser.add_argument("-s", "--saving_path", default='./', help='path to location where cropped patches will be saved')
    parser.add_argument("-t", "--text", action='store_true', help='flag for generating a text file containing file '
                                                                  'names of all cropped image patches from video')
    parser.add_argument("-f", "--output_format", default='.png', help='output image patch format')
    args = parser.parse_args()

    # initialize global environment variables
    current_class = 0
    current_frame = 0
    video_path = args.file_path
    num_class = args.num_class
    if args.class_prefix:
        if num_class == len(args.class_prefix):
            class_prefix = args.class_prefix
        else:
            raise RuntimeError('Number of prefixes given in the input argument (%d) is inconsistent with number of '
                               'classes (%d) specified' % (len(args.class_prefix), num_class))
    else:
        class_prefix = ['class'+'0'*(int(np.log10(num_class)) - int(np.log10(i+0.9)))+str(i)+'_patch' for i in range(
            num_class)]
    saving_path = args.saving_path
    if args.text:
        file_name_list = [[] for i in range(num_class)]
    save_file_names = args.text
    output_format = args.output_format
    n_patch_saved = [0 for i in range(num_class)]  # number of patches that has been cropped and saved for each class
    x_list = []  # x-coordinate of clicked point of mouse in image (column index)
    y_list = []  # y-coordinate of clicked point of mouse in image (row index)
    mouse_clicked = False
    frame_copy_old, frame_copy_new = None, None
    frame = None
    class_color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
                   [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0], [0, 128, 128], [128, 0, 128]]
    print('VideoCropper v0.1\n'+'-='*64)


def click_and_draw_box(event, x, y, flag, param):
    global current_class, current_frame, saving_path, class_prefix, file_name_list, output_format, n_patch_saved, \
        mouse_clicked, x_list, y_list, frame_copy_old, frame_copy_new, frame, class_color
    if event == cv2.EVENT_LBUTTONUP:
        if not mouse_clicked:  # if mouse is clicked for the first time, record clicked location
            x_list.append(x)
            y_list.append(y)
            mouse_clicked = True
        else:  # if mouse is clicked for the second time, draw rectangle based on two clicks and ask save or discard
            if len(x_list) == 1:
                x_list.append(x)
                y_list.append(y)
                top_left_corner = (min(x_list), min(y_list))
                bot_right_corner = (max(x_list), max(y_list))
                frame_copy_new = frame_copy_old.copy()
                cv2.rectangle(frame_copy_new, top_left_corner, bot_right_corner, class_color[current_class])
                cv2.imshow('frame', frame_copy_new)
                print('Press s to save or press d to discard selected patch')
                while True:  # wait for either 's' or 'd' pressed, ignore all other keyboard input
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('s'):  # save the selected patch and update environment variables
                        file_name = saving_path + class_prefix[current_class] + \
                                    '0'*(5-int(np.log10(n_patch_saved[current_class]+0.9))) + \
                                    str(n_patch_saved[current_class]) + output_format
                        file_name_list[current_class].append(file_name)
                        frame_copy_old = frame_copy_new
                        cv2.imwrite(file_name, frame[top_left_corner[1]:bot_right_corner[1], top_left_corner[0]:bot_right_corner[0], :])
                        print('Cropped patch saved as ' + file_name)
                        n_patch_saved[current_class] += 1
                        print('%d patches cropped and saved for class %s' % (n_patch_saved[current_class],
                                                                             class_prefix[current_class]))
                        x_list.clear()  # reset coordinates and mouse click status
                        y_list.clear()
                        mouse_clicked = False
                        break
                    elif key == ord('d'):  # discard the selected patch and show old frame
                        cv2.imshow('frame', frame_copy_old)
                        x_list.clear()
                        y_list.clear()
                        mouse_clicked = False
                        break


def main():
    global current_class, current_frame, video_path, num_class, saving_path, class_prefix, file_name_list, frame, \
        save_file_names, output_format, n_patch_saved, mouse_clicked, x_list, y_list, frame_copy_old, frame_copy_new
    video_object = cv2.VideoCapture(video_path)
    if not video_object.isOpened():
        raise RuntimeError('Unable to open file'+video_path)
    retval, frame = video_object.read()
    frame_copy_old = frame
    if not retval:
        raise RuntimeError('Unable to grab next frame')
    current_frame = int(video_object.get(propId=1))-1
    total_frame_num = int(video_object.get(propId=7))
    print('Video file: ' + video_path + '\t (Total number of frames: %d' % (total_frame_num,))
    print('Current frame number: %d\t ---Current class set to %s\t' % (current_frame, class_prefix[current_class]))
    print('Cropped patches will be saved to ' + saving_path)
    print('Prefix of different class of patches are:', end=' ')
    print(*class_prefix, sep=', ')
    print('Cropped patches will be saved as %s file' % (output_format,))
    cv2.namedWindow('frame', flags=1)
    cv2.imshow('frame', frame)
    cv2.setMouseCallback('frame', click_and_draw_box)
    print('Press n to go to next frame. Press p to go to previous frame. Press q to quit program.')
    print('Press a number to change selected class to that number.')
    print('Press w to write the current displayed frame to an image file. Saved frame will be under the same directory '
          'as cropped patches and will have same image format')
    print('Press i for frame number and current selected class info')
    print('Press h for help on hot keys')
    print('Click on two points to crop a rectangular patch\n\n')

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):
            if not retval:
                print('This is the last frame of the video sequence being cropped. Cannot go to next frame')
            else:
                retval, frame = video_object.read()
                frame_copy_old = frame
                x_list.clear()
                y_list.clear()
                mouse_clicked = False
                current_frame += 1
                cv2.imshow('frame', frame)
                print('Current frame number: %d' % (current_frame,))
        elif key == ord('p'):
            if not current_frame:
                print('This is the first frame of the video sequence being cropped. Cannot go to previous frame')
            else:
                current_frame -= 1
                if not video_object.set(propId=1, value=current_frame):
                    raise RuntimeError('Fail to go to previous frame')
                retval, frame = video_object.read()
                frame_copy_old = frame
                x_list.clear()
                y_list.clear()
                mouse_clicked = False
                cv2.imshow('frame', frame)
                print('Current frame number: %d' % (current_frame,))
        elif key == ord('q'):
            if input('Are you sure about exiting the program? [y/n]: ') == 'y':
                print('Terminating')
                break
        elif key == ord('w'):
            frame_name = saving_path + 'frame_' + '0'*(int(np.log10(total_frame_num)) -
                                                       int(np.log10(current_frame+0.9))) + \
                         str(current_frame) + output_format
            cv2.imwrite(frame_name, frame_copy_old)
            print('Current displayed frame has been writen to ' + frame_name)
        elif key >= ord('0') and key <= ord('9'):
            current_class = key - ord('0')
            print('Set selected class to %s' % (class_prefix[current_class],))
        elif key == ord('h'):
            print('Press n to go to next frame. Press p to go to previous frame. Press q to quit program.')
            print('Press a number to change selected class to that number.')
            print('Press w to write the current displayed frame to an image file. Saved frame will be under the same '
                  'directory as cropped patches and will have same image format')
            print('Press i for frame number and current selected class info')
            print('Press h for help on hot keys')
            print('Click on two points to crop a rectangular patch')
        elif key == ord('i'):
            print('Current frame number: %d' % (current_frame,))
            print('Current selected class is class %s\n' % (class_prefix[current_class],))

    if save_file_names:
        f_name = saving_path + 'frame_patch_names.txt'
        if os.path.exists(f_name):
            if input('File %s already exists. Continue saving frame patch names will overwrite that file. Do you want '
                     'to continue? [y/n]' %(f_name,)) != 'y':
                print('\nExit')
                exit(0)
        with open(f_name, 'w') as fh:
            for class_k_filenames in file_name_list:
                fh.writelines('%s\n' % patch_name for patch_name in class_k_filenames)
        print('File %s writen and closed' % (f_name))


if __name__ == '__main__':
    main()
