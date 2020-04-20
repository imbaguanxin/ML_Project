import os

from PIL import Image
import matplotlib.pyplot as plt


def tosaka():
    tosaka1 = Image.open(os.path.join('project_document', 'images', 'tosaka1.jpg'))
    tosaka1 = tosaka1.crop((25, 25, 350, 350))
    tosaka2 = Image.open(os.path.join('project_document', 'images', 'tosaka2.jpg'))
    tosaka2 = tosaka2.crop((50, 50, 650, 650))
    tosaka3 = Image.open(os.path.join('project_document', 'images', 'tosaka3.png'))
    tosaka3 = tosaka3.crop((100, 100, 700, 700))
    tosaka4 = Image.open(os.path.join('project_document', 'images', 'tosaka4.jpg'))
    tosaka4 = tosaka4.crop((400, 100, 1300, 1000))

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    plt.imshow(tosaka1)
    plt.axis('off')
    plt.title('Tosaka Rin by sogamakoto')

    ax2 = fig.add_subplot(gs[0, 1])
    plt.imshow(tosaka2)
    plt.axis('off')
    plt.title('Tosaka Rin by Lpip')

    ax3 = fig.add_subplot(gs[1, 0])
    plt.imshow(tosaka3)
    plt.axis('off')
    plt.title('Tosaka Rin by SUEUN')

    ax3 = fig.add_subplot(gs[1, 1])
    plt.imshow(tosaka4)
    plt.axis('off')
    plt.title('Tosaka Rin by Azomo')
    plt.savefig('project_document/images/tosaka_grid.png', dpi=350)
    plt.show()


def saber_like():
    saber = Image.open(os.path.join('project_document', 'images', 'saber.jpg'))
    saber = saber.crop((100, 150, 450, 500))
    lily = Image.open(os.path.join('project_document', 'images', 'lily.jpg'))
    lily = lily.crop((50, 100, 500, 500))
    nero = Image.open(os.path.join('project_document', 'images', 'nero.jpg'))
    nero = nero.crop((250, 100, 650, 500))
    alter = Image.open(os.path.join('project_document', 'images', 'alter.jpg'))
    alter = alter.crop((250, 100, 550, 400))

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    plt.imshow(saber)
    plt.axis('off')
    plt.title('saber')

    ax2 = fig.add_subplot(gs[0, 1])
    plt.imshow(lily)
    plt.axis('off')
    plt.title('lily')

    ax3 = fig.add_subplot(gs[1, 0])
    plt.imshow(nero)
    plt.axis('off')
    plt.title('nero')

    ax3 = fig.add_subplot(gs[1, 1])
    plt.imshow(alter)
    plt.axis('off')
    plt.title('alter')
    plt.savefig('project_document/images/saber_grid.png', dpi=350)
    plt.show()


if __name__ == '__main__':
    saber_like()
