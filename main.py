"""
Модуль main для реализации сжатия изображений с помощью квадратных деревьев.
"""
import argparse
import os

from PIL import Image

from QuadtreeImage import QuadTree


def compress_image(image_path: str, depth: int, show_lines: bool) -> None:
    """
    Сжимает изображение с использованием квадродерева.

    Аргументы:
        image_path (str): Путь к файлу входного изображения.
        depth (int): Глубина рекурсивного поиска в квадродереве.
        show_lines (bool): Показывать линии на выходном изображении.
    """
    image = Image.open(image_path)
    filename = os.path.splitext(os.path.basename(image_path))[0]

    quadtree = QuadTree(image)
    output_image = quadtree.create_image(depth, show_lines=show_lines)
    quadtree.create_gif(f"compressed_{filename}.gif", show_lines=show_lines)
    output_image.save(f"compressed_{filename}.jpg")


def main() -> None:
    parser = argparse.ArgumentParser(description="Создание изображения с использованием квадродерева")
    parser.add_argument("image_path", type=str, help="Путь к файлу входного изображения")
    parser.add_argument("--depth", "-d", type=int, default=1, help="Глубина рекурсивного поиска в квадродереве")
    parser.add_argument("--show_lines", "-s", action="store_true", help="Показывать линии на выходном изображении")

    args = parser.parse_args()
    compress_image(args.image_path, args.depth, args.show_lines)


if __name__ == '__main__':
    main()

# python main.py spanch_bob.jpg -d 8 -s
# python main.py Shakal4k.png -d 5 -s