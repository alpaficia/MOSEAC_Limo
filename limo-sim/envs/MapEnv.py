import numpy as np
import matplotlib.path as mplPath


def direction(x1, y1, x2, y2, x3, y3):
    return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)


def line_intersection(line1, line2):
    """
    计算两条线段的交点，如果没有交点返回None。
    线段重合的情况也会被考虑。
    """

    # 定义线段的方程：P = P0 + t * (P1 - P0)
    def line(p0, p1, t):
        return p0 + t * (p1 - p0)

    # 计算线性方程的参数 t 和 u
    def compute_t_u(line1, line2):
        x1, y1, x2, y2 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
        x3, y3, x4, y4 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None  # 线段平行或重合
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
        return t, u

    result = compute_t_u(line1, line2)
    if result:
        t, u = result
        if 0 <= t <= 1 and 0 <= u <= 1:
            # 两线段相交
            intersection_point = line(np.array(line1[0]), np.array(line1[1]), t)
            return intersection_point
        else:
            return None


def point_to_line_distance(point, line):
    """
    Calculate the minimum distance from a point to a line segment.

    Parameters:
    - point: A numpy array with shape (2,), representing the (x, y) coordinates of the point.
    - line: A numpy array with shape (2,2), representing the endpoints of the line segment.

    Returns:
    - The minimum distance from the point to the line segment.
    """
    # Vector from line start to point
    line_start_to_point = point - line[0]
    # Vector from line start to line end
    line_vector = line[1] - line[0]
    # Project vector from line start to point onto the line vector
    line_vector_norm = np.linalg.norm(line_vector)
    if line_vector_norm == 0:
        # The line start and end points are the same
        return np.linalg.norm(line_start_to_point)
    line_unit_vector = line_vector / line_vector_norm
    projection_length = np.dot(line_start_to_point, line_unit_vector)
    if projection_length < 0:
        # Closest point is the line start
        return np.linalg.norm(line_start_to_point)
    elif projection_length > line_vector_norm:
        # Closest point is the line end
        return np.linalg.norm(point - line[1])
    else:
        # Closest point is on the segment
        closest_point_on_segment = line[0] + projection_length * line_unit_vector
        return np.linalg.norm(point - closest_point_on_segment)


def closest_distance_to_walls(point, walls):
    # 这里需要一个实现，计算点到所有墙体的最短距离
    min_distance = np.inf
    for wall in walls:
        distance = np.linalg.norm(np.cross(wall[1]-wall[0], wall[0]-point)) / np.linalg.norm(wall[1]-wall[0])
        min_distance = min(min_distance, distance)
    return min_distance


class MapINF3995:
    # define the map region by yourself
    def __init__(self, size=(-1.5, 1.5), interval=0.1):
        self.safe_distance = 0.10
        self.size = size
        self.wall1 = np.array([[0.0, 0.0], [-1.0, 0.0]])
        self.wall2 = np.array([[-1.0, 0.0], [-1.0, -1.0]])
        self.wall3 = np.array([[-1.0, -1.0], [0.0, -1.0]])
        self.wall4 = np.array([[0.0, -1.0], [0.0, -0.75]])
        self.wall5 = np.array([[0.0, -0.75], [-0.57, -0.75]])
        self.wall6 = np.array([[-0.57, -0.75], [-0.57, -0.3]])
        self.wall7 = np.array([[-0.57, -0.3], [0.0, -0.3]])
        self.wall8 = np.array([[0.0, -0.3], [0.0, 0.0]])
        self.zone_left_down = mplPath.Path(np.vstack([self.wall1, self.wall2, self.wall3, self.wall4, self.wall5,
                                                      self.wall6, self.wall7, self.wall8]))
        # 左下

        self.wall9 = np.array([[0.0, 0.4], [0.0, 1.0]])
        self.wall10 = np.array([[0.0, 1.0], [-1.0, 1.0]])
        self.wall11 = np.array([[-1.0, 1.0], [-1.0, 0.4]])
        self.wall12 = np.array([[-1.0, 0.4], [0.0, 0.4]])
        self.zone_left_up = mplPath.Path(np.vstack([self.wall9, self.wall10, self.wall11, self.wall12]))
        # 左上

        self.wall13 = np.array([[0.5, 0.0], [1.0, 0.0]])
        self.wall14 = np.array([[1.0, 0.0], [1.0, 1.0]])
        self.wall15 = np.array([[1.0, 1.0], [0.5, 1.0]])
        self.wall16 = np.array([[0.5, 1.0], [0.5, 0.0]])
        self.zone_right_up = mplPath.Path(np.vstack([self.wall13, self.wall14, self.wall15, self.wall16]))
        # 右上

        self.wall17 = np.array([[0.5, -1.0], [0.5, -0.5]])
        self.wall18 = np.array([[0.5, -0.5], [1.0, -0.5]])
        self.wall19 = np.array([[1.0, -0.5], [1.0, -1.0]])
        self.wall20 = np.array([[1.0, -1.0], [0.5, -1.0]])
        self.zone_right_down = mplPath.Path(np.vstack([self.wall17, self.wall18, self.wall19, self.wall20]))
        # 右下

        self.wall21 = np.array([[1.4, -1.2], [1.2, -1.4]])
        self.wall22 = np.array([[1.2, -1.4], [-1.2, -1.4]])
        self.wall23 = np.array([[-1.2, -1.4], [-1.4, -1.2]])
        self.wall24 = np.array([[-1.4, -1.2], [-1.4, 1.2]])
        self.wall25 = np.array([[-1.4, 1.2], [-1.2, 1.4]])
        self.wall26 = np.array([[-1.2, 1.4], [1.2, 1.4]])
        self.wall27 = np.array([[1.2, 1.4], [1.4, 1.2]])
        self.wall28 = np.array([[1.4, 1.2], [1.4, -1.2]])
        # 外墙
        self.walls = np.array(
            [self.wall1, self.wall2, self.wall3, self.wall4, self.wall5, self.wall6, self.wall7, self.wall8, self.wall9,
             self.wall10, self.wall11, self.wall12, self.wall13, self.wall14, self.wall15, self.wall16, self.wall17,
             self.wall18, self.wall19, self.wall20, self.wall21, self.wall22, self.wall23, self.wall24, self.wall25,
             self.wall26, self.wall27, self.wall28])
        self.out_walls = np.array([self.wall21, self.wall22, self.wall23, self.wall24, self.wall25, self.wall26,
                                   self.wall27, self.wall28])
        self.inner_walls = np.array(
            [self.wall1, self.wall2, self.wall3, self.wall4, self.wall5, self.wall6, self.wall7, self.wall8, self.wall9,
             self.wall10, self.wall11, self.wall12, self.wall13, self.wall14, self.wall15, self.wall16, self.wall17,
             self.wall18, self.wall19, self.wall20])

    def radar_intersections(self, point):
        distances = []
        intersections = []
        # 生成20条射线
        for angle in np.linspace(0, 2 * np.pi, 20, endpoint=False):
            # 射线方向向量
            ray_dir = np.array([np.cos(angle), np.sin(angle)])
            ray_start = point
            # 假设射线长度为一段很大的距离，这里设置为1000个单位
            ray_end = ray_start + ray_dir * 1000
            ray_line = [ray_start, ray_end]

            # 记录最近的交点距离和坐标
            nearest_distance = float('inf')
            nearest_intersection = np.array([np.nan, np.nan])  # 默认为np.nan
            for wall in self.walls:
                intersection = line_intersection(ray_line, wall)
                if intersection is not None:
                    # 如果找到交点，计算雷达到交点的距离
                    distance = np.linalg.norm(intersection - ray_start)
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_intersection = intersection

            # 如果找到交点，添加最近交点的距离和坐标
            if not np.isnan(nearest_intersection).any():
                distances.append(nearest_distance)
            else:
                distances.append(np.nan)  # 没有交点则距离为np.nan
            intersections.append(nearest_intersection)

        return np.array(intersections)

    def is_collision_with_inner_wall(self, point1, point2):
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]

        for wall in self.inner_walls:
            x3, y3 = wall[0][0], wall[0][1]
            x4, y4 = wall[1][0], wall[1][1]
            if ((direction(x1, y1, x2, y2, x3, y3) * direction(x1, y1, x2, y2, x4, y4) < 0) and
                    (direction(x3, y3, x4, y4, x1, y1) * direction(x3, y3, x4, y4, x2, y2) < 0)):
                return True  # 直接在检测到碰撞时返回True
        return False  # 如果遍历完所有内墙都没有发现碰撞，则返回False

    def is_collision_with_out_wall(self, point1, point2):
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]

        for wall in self.out_walls:
            x3, y3 = wall[0][0], wall[0][1]
            x4, y4 = wall[1][0], wall[1][1]
            if ((direction(x1, y1, x2, y2, x3, y3) * direction(x1, y1, x2, y2, x4, y4) < 0) and
                    (direction(x3, y3, x4, y4, x1, y1) * direction(x3, y3, x4, y4, x2, y2) < 0)):
                return True  # 直接在检测到碰撞时返回True
        return False  # 如果遍历完所有内墙都没有发现碰撞，则返回False

    def get_wall(self):
        return np.array(self.walls)

    def generate_point_outside_region(self):
        while True:
            x, y = np.random.uniform(self.size[0], self.size[1]), np.random.uniform(self.size[0], self.size[1])
            point = np.array([x, y])
            zones = [self.zone_left_down, self.zone_left_up, self.zone_right_up, self.zone_right_down]
            dis2wall = closest_distance_to_walls(point, self.inner_walls)

            if all(not zone.contains_point(point) for zone in zones) and dis2wall >= self.safe_distance:
                return point
