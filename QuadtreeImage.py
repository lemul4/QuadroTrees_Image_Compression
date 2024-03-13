"""
Модуль QuadtreeImage с реализацией квадродерева.
"""
import threading
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


class Quadrant:
    """
    Представляет квадрант в квадродереве.

    Атрибуты:
        bbox (Tuple[int, int, int, int]): Границы квадранта.
        depth (int): Глубина в квадродереве.
        children (Optional[List[Quadrant]]): Дочерние квадранты, если они есть.
        leaf (bool): True, если квадрант является листом, иначе False.
        y (float): Яркость.
        color (Tuple[int, int, int]): Средний цвет в формате RGB.
    """

    def __init__(self, image: Image.Image, bbox: Tuple, depth: int):
        """
        Инициализирует квадрант.

        Аргументы:
            image (PIL.Image.Image): Входное изображение.
            bbox (Tuple[int, int, int, int]): Границы квадранта.
            depth (int): Глубина в квадродереве.
        """
        self.bbox = bbox
        self.depth = depth
        self.children: Optional[List[Quadrant]] = None
        self.leaf = False
        self.y, self.color = self.calculate_properties(image)

    def calculate_properties(self, image: Image.Image) -> tuple[float, tuple[int, ...]]:
        """
        Рассчитывает яркость (Y) и средний цвет квадранта.

        Аргументы:
            image (PIL.Image.Image): Входное изображение.

        Возвращает:
            Tuple[float, Tuple[int, int, int]]: Яркость (Y) и средний цвет.
        """
        image = image.crop(self.bbox)
        histogram = image.histogram()

        y = self.calculate_y(histogram)
        color = self.calculate_average_color(image)

        return y, color

    def split(self, image: Image.Image):
        """
        Разделяет квадрант на четыре новых квадранта.

        Аргументы:
            image (PIL.Image.Image): Входное изображение.
        """
        left, upper, width, height = self.bbox
        middle_x = left + (width - left) / 2
        middle_y = upper + (height - upper) / 2

        upper_left = Quadrant(image, (left, upper, middle_x, middle_y), self.depth + 1)
        upper_right = Quadrant(image, (middle_x, upper, width, middle_y), self.depth + 1)
        bottom_left = Quadrant(image, (left, middle_y, middle_x, height), self.depth + 1)
        bottom_right = Quadrant(image, (middle_x, middle_y, width, height), self.depth + 1)

        self.children = [upper_left, upper_right, bottom_left, bottom_right]

    @staticmethod
    def calculate_average_color(image: Image.Image) -> tuple[int, ...]:
        """
        Рассчитывает средний цвет изображения.

        Аргументы:
            image (PIL.Image.Image): Входное изображение.

        Возвращает:
            Tuple[int, int, int]: Средний цвет в формате RGB.
        """
        image_arr = np.asarray(image)
        avg_color_per_row = np.average(image_arr, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        return tuple(map(round, avg_color))

    @staticmethod
    def calculate_y(hist: List[int]) -> float:
        """
        Рассчитывает Y-компоненту в модели цвета YUV.

        Аргументы:
            hist (List[int]): Данные гистограммы.

        Возвращает:
            float: Y-компонента.
        """
        red_deviation = Quadrant.standard_deviation(hist[:256])
        green_deviation = Quadrant.standard_deviation(hist[256:512])
        blue_deviation = Quadrant.standard_deviation(hist[512:768])

        y = red_deviation * 0.2989 + green_deviation * 0.587 + blue_deviation * 0.114
        return y

    @staticmethod
    def standard_deviation(hist: List[int]) -> float:
        """
        Рассчитывает стандартное отклонение гистограммы изображения.

        Аргументы:
            hist (List[int]): Данные гистограммы.

        Возвращает:
            float: Значение стандартного отклонения.
        """
        total = sum(hist)
        value, deviation_number = 0, 0

        if total > 0:
            value = sum(i * x for i, x in enumerate(hist)) / total
            deviation_number = (sum(x * (value - i) ** 2 for i, x in enumerate(hist)) / total) ** 0.5

        return deviation_number


class QuadTree:
    """
    Представляет квадродерево для сжатия изображения.

    Атрибуты:
        width (int): Ширина входного изображения.
        height (int): Высота входного изображения.
        max_depth (int): Максимальная глубина квадродерева.
        root (Quadrant): Корневой квадрант квадродерева.
    """

    def __init__(self, image: Image.Image):
        """
        Инициализирует квадродерево.

        Аргументы:
            image (PIL.Image.Image): Входное изображение для квадродерева.
        """
        self.root = None
        self.width, self.height = image.size
        self.max_depth = 0
        self.start(image)

    def start(self, image: Image.Image):
        """
        Начинает сжатие квадродеревом.

        Аргументы:
            image (PIL.Image.Image): Входное изображение.
        """
        self.root = Quadrant(image, image.getbbox(), 0)
        self.build(self.root, image)

    def build(self, root: Quadrant, image: Image.Image):
        """
        Рекурсивно строит квадродерево.

        Аргументы:
            root (Quadrant): Корневой квадрант.
            image (PIL.Image.Image): Входное изображение.
        """
        if root.depth >= 8 or root.y <= 10:
            if root.depth > self.max_depth:
                self.max_depth = root.depth
            root.leaf = True
            return

        root.split(image)

        if root.depth == 0:
            threads = []
            for child in root.children:
                thread = threading.Thread(target=self.build, args=(child, image))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
        else:
            for children in root.children:
                self.build(children, image)

    def create_image(self, custom_depth: int, show_lines: bool = False) -> Image.Image:
        """
        Создает изображение на основе сжатия квадродеревом.

        Аргументы:
            custom_depth (int): Глубина рекурсивного поиска.
            show_lines (bool): Показывать ли линии на выходном изображении.

        Возвращает:
            PIL.Image.Image: Выходное изображение.
        """
        image = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(image)
        leaf_quadrants = self.get_leaves(custom_depth)

        for quadrant in leaf_quadrants:
            if show_lines:
                draw.rectangle(quadrant.bbox, quadrant.color, outline=(0, 0, 0))
            else:
                draw.rectangle(quadrant.bbox, quadrant.color)

        return image

    def get_leaves(self, depth: int) -> List[Quadrant]:
        """
        Получает список листьев дерева.

        Аргументы:
            depth (int): Глубина в квадродереве.

        Возвращает:
            List[Quadrant]: Список листьев.
        """
        if depth > self.max_depth:
            depth = self.max_depth

        quadrants = []
        self.search(self.root, depth, quadrants.append)

        return quadrants

    def search(self, quadrant: Quadrant, max_depth: int, append_leaf: Callable[[Quadrant], None]):
        """
        Рекурсивно ищет в квадродереве листья или квадранты максимальной глубины.

        Аргументы:
            quadrant (Quadrant): Текущий квадрант.
            max_depth (int): Максимальная глубина.
            append_leaf (Callable[[Quadrant], None]): Функция для добавления листового квадранта
                в список quadrants в get_leaves.
        """
        if quadrant.leaf or quadrant.depth == max_depth:
            append_leaf(quadrant)
        elif quadrant.children is not None:
            for child in quadrant.children:
                self.search(child, max_depth, append_leaf)

    def create_gif(self, file_name: str, duration: int = 1000, loop: int = 0, show_lines: bool = False):
        """
        Создает анимацию GIF сжатия квадродеревом.

        Аргументы:
            file_name (str): Имя выходного файла GIF.
            duration (int): Длительность каждого кадра в миллисекундах.
            loop (int): Количество циклов.
            show_lines (bool): Показывать ли линии на выходном изображении.
        """
        gif = []

        for i in range(self.max_depth):
            image = self.create_image(i, show_lines=show_lines)
            gif.append(image)
        if gif:
            gif.reverse()
            gif[0].save(
                file_name,
                save_all=True,
                append_images=gif[1:],
                duration=duration, loop=loop)
        else:
            print('Не удалось создать GIF')
