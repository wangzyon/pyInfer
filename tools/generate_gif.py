import imageio


def create_gif(image_list, gif_name, duration=1.0):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间
    :return:
    '''
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))

    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def main():
    import os
    image_dir = "C:/Users/wzy/Desktop/NOTEBOOK/docs/algorithm/images/新建文件夹"
    image_list = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
    gif_name = os.path.join(image_dir, "1.gif")
    duration = 1.5
    create_gif(image_list, gif_name, duration)


if __name__ == '__main__':
    main()