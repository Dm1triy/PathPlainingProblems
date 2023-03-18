import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import collections as mc
import math
from rdp import rdp
import cv2 as cv

from Dijkstra import Dijkstra
from A_star import Astar
from RRT_star import Rrtstar


class MapPlotter:
    def __init__(self, log='examp4.txt'):
        self.cell_size = 0.1  # m
        self.map_width = 251
        self.map_height = 251
        self.map = np.array([[0] * self.map_width]*self.map_height)
        self.start_x = self.map_width//2
        self.start_y = self.map_height//2
        self.log_data = self.read_log(log)

        self.alg = None

    @staticmethod
    def read_log(log):
        f = open(log, "r")
        log_lines = f.read().split("\n")
        data = []
        for i in log_lines:
            if i == '':
                break
            temp_data = i.split(';')
            odom = list(map(float, temp_data[0].split(',')))
            lidar = list(map(float, temp_data[1].split(',')))
            data.append([odom, lidar])
        return data

    @staticmethod
    def reshape_array(array):
        """
        Convert [[i1, j1], [i2, j2], ...] into [[i1, i2, ...], [j1, j1, ...]]
        :param: array [[i1, j1], [i2, j2], ...]
        :return: array [[i1, i2, ...], [j1, j1, ...]]
        """
        array = np.array(array)
        array = np.array(np.split(array, 2, axis=1))
        array = array.reshape(array.shape[0], array.shape[1])
        return array

    def print_map(self, robot, direction=None):
        fig, ax = plt.subplots(figsize=(15, 8))
        cmap = colors.ListedColormap(['White', 'Blue'])
        ax.pcolor(self.map, cmap=cmap, snap=True)

        ax.set_xlim(self.start_x, self.map_width)  # ax.set_xlim(0, self.map_width)
        ax.set_ylim(0, self.map_height)
        ax.invert_yaxis()
        if direction:
            ax.quiver(robot[0] + 0.5, robot[1] + 0.5, direction[0], direction[1], color='r')
        ax.scatter(robot[0] + 0.5, robot[1] + 0.5, marker='P')

        majorx_ticks = np.arange(self.start_x, self.map_width-1, 10)    # majorx_ticks = np.arange(0, self.map_width-1, 10)
        minorx_ticks = np.arange(self.start_x, self.map_width-1, 1)     # minorx_ticks = np.arange(0, self.map_width-1, 1)
        majory_ticks = np.arange(0, self.map_height-1, 10)
        minory_ticks = np.arange(0, self.map_height-1, 1)

        x_lim = (self.map_width-1) * self.cell_size/2
        y_lim = (self.map_height-1) * self.cell_size/2
        majorx_labels = np.arange(0, x_lim, 10 * self.cell_size)    # majorx_labels = np.arange(-x_lim, x_lim, 10 * self.cell_size)

        majory_labels = np.arange(-y_lim, y_lim, 10 * self.cell_size)
        majory_labels = majory_labels * -1

        ax.set_xticks(majorx_ticks, labels=majorx_labels)
        ax.set_xticks(minorx_ticks, labels=[], minor=True)
        ax.set_yticks(majory_ticks, labels=majory_labels)
        ax.set_yticks(minory_ticks, labels=[], minor=True)

        # ax.set_xticks(majorx_ticks)
        # ax.set_xticks(minorx_ticks, minor=True)
        # ax.set_yticks(majory_ticks)
        # ax.set_yticks(minory_ticks, minor=True)

        ax.grid(which='Both')

        fig.tight_layout()
        plt.show()

    def pos_to_cell(self, x, y):
        return int(self.start_x + x/self.cell_size), int(self.start_y - y/self.cell_size)

    def cell_to_pos(self, x, y):
        # x positive if index is higher (right); y positive if index is smaller (up)
        return np.round((x - self.start_x) * self.cell_size, 2), \
               np.round((self.start_y - y) * self.cell_size, 2)

    def obstacle_indexes(self, robot_pos, robot_ang, lidar):
        obs_indx = []
        for i in range(len(lidar)):
            if lidar[i] > 5 or lidar[i] < 0.5:
                continue
            ang = math.radians(240)/len(lidar) * i - robot_ang - math.radians(120)

            # lidar radius in meter therefore radius*cos(ang) should be converted to the index
            # 0.3 * math.cos(robot_ang) and 0.3 * math.sin(robot_ang) are x and y of the Lidar in front of the robot
            local_indx = 0.3 * math.cos(robot_ang) + lidar[i] * math.cos(ang)/self.cell_size,\
                         0.3 * math.sin(robot_ang) + lidar[i] * math.sin(ang)/self.cell_size
            obs_indx.append([int(robot_pos[0] + local_indx[0]), int(robot_pos[1] + local_indx[1])])
        obs_indx = self.reshape_array(obs_indx)
        return obs_indx

    def map_in_time(self, index):
        cur_data = self.log_data[index]
        robot_x, robot_y, robot_ang = cur_data[0]

        robot_indx = self.pos_to_cell(robot_x, robot_y)
        robot_dir = math.cos(robot_ang), math.sin(robot_ang)

        obstacle_indx = self.obstacle_indexes(robot_indx, robot_ang, cur_data[1])
        self.map[obstacle_indx[1], obstacle_indx[0]] = 1
        return robot_indx, robot_dir

    # Ramer–Douglas–Peucker alg (not used anywhere)
    def reduce_map(self):
        obs_indexes = np.where(self.map == 1)
        self.map[obs_indexes[0], obs_indexes[1]] = 0
        obs_indexes = np.stack(obs_indexes, axis=1)

        new_indexes = rdp(obs_indexes, epsilon=0)
        new_indexes = self.reshape_array(new_indexes)
        self.map[new_indexes[0], new_indexes[1]] = 1

    def final_map(self):
        trajectory = []
        robot_dir = []
        for i in range(len(self.log_data)):
            robot_pos, robot_dir = self.map_in_time(i)
            trajectory.append(robot_pos)
        self.print_map(trajectory[-1], robot_dir)
        self.print_processed_map(trajectory)

    def print_processed_map(self, trajectory):
        img = np.zeros((self.map_width, self.map_height, 3), np.uint8)
        t_map = np.array(self.map, np.uint8) * 255

        lines = cv.HoughLinesP(t_map, 2, np.pi/180, 59, 7, 6)
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        scale = 4
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        threshold = 10
        canny = cv.Canny(gray, threshold, threshold * 2)
        contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # contour printing
        map_contours = np.zeros((canny.shape[0], canny.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            cv.drawContours(map_contours, contours, i, (255, 0, 0))

        hull = []
        for i in range(len(contours)):
            hull.append(cv.convexHull(contours[i]))

        map_hull = np.zeros((canny.shape[0], canny.shape[1], 3), dtype=np.uint8)
        for i in range(len(contours)):
            cv.drawContours(map_hull, hull, i, (255, 0, 0))

        # print the robot trajectory
        for i in range(len(trajectory)):
            center = (int(trajectory[i][0] * scale), int(trajectory[i][1] * scale))
            cv.circle(map_contours, center, 3, (0, 0, 255))
            cv.circle(map_hull, center, 3, (0, 0, 255))

        # cv.imshow('borders', map_hull)
        # cv.waitKey(0)
        # cv.imshow('borders', map_contours)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # rewrite the map with the processed map
        map_contours = cv.resize(img, (self.map_width, self.map_height), interpolation=cv.INTER_AREA)
        map_contours = np.amax(map_contours, axis=2)
        map_contours = np.where(map_contours == 255, 1, 0)
        self.map = map_contours
        self.print_map(trajectory[0])

    def path_planning_sim(self):
        # start_point = 160, 160
        # end_point = 125, 195
        start_point = 160, 160
        end_point = 195, 125
        space = self.map.copy()
        # self.alg = Dijkstra(start_point=np.array(start_point),
        #                     end_point=np.array(end_point), bin_map=space)
        # self.alg = Astar(start_point=np.array(start_point),
        #                  end_point=np.array(end_point), bin_map=space)
        self.alg = Rrtstar(start_point=np.array(start_point),
                           end_point=np.array(end_point), bin_map=space)
        ax, fig = self.init_plot(start_point, end_point, space)

        i = 0
        line_segments = None
        while not self.alg.dist_reached:
            self.alg.step()
            if i % 200 == 0:
                node_indx = self.reshape_array(self.alg.nodes)
                ax.scatter(node_indx[0]+0.5, node_indx[1]+0.5, color='y', s=2, marker='H')
                if line_segments:
                    mc.LineCollection.remove(line_segments)
                line_segments = self.display_connections()
                ax.add_collection(line_segments)
                plt.pause(0.1)
            i += 1
        path = self.alg.get_path()
        if not path:
            return
        path = self.reshape_array(path)
        # ax.scatter(path[0]+0.5, path[1]+0.5, color='r', s=10, marker='H')
        ax.plot(path[0] + 0.5, path[1] + 0.5, color='r')
        plt.show()

    def display_connections(self):
        lines = []
        if not self.alg.graph:
            return
        for i in self.alg.graph:
            node = self.alg.graph[i]
            if node.parent:
                parent = self.alg.graph[node.parent]
                lines.append([(node.x+0.5, node.y+0.5), (parent.x+0.5, parent.y+0.5)])
        color = (0, 0, 0, 1)
        lc = mc.LineCollection(lines, colors=color, linewidths=1)
        return lc

    def init_plot(self, start_point, end_point, space):
        start = plt.Circle(start_point, 1, color='r')
        end = plt.Circle(end_point, 1, color='r')
        fig, ax = plt.subplots(figsize=(15, 8))
        cmap = colors.ListedColormap(['White', 'Blue'])
        ax.pcolor(space, cmap=cmap, snap=True)
        ax.add_patch(start)
        ax.add_patch(end)
        ax.set_xlim(self.start_x, self.map_width)
        ax.set_ylim(0, self.map_height)
        ax.invert_yaxis()

        majorx_ticks = np.arange(self.start_x, self.map_width - 1, 10)
        minorx_ticks = np.arange(self.start_x, self.map_width - 1, 1)
        majory_ticks = np.arange(0, self.map_height - 1, 10)
        minory_ticks = np.arange(0, self.map_height - 1, 1)

        ax.set_xticks(majorx_ticks)
        ax.set_xticks(minorx_ticks, minor=True)
        ax.set_yticks(majory_ticks)
        ax.set_yticks(minory_ticks, minor=True)
        ax.grid(which='Both')
        fig.tight_layout()
        return ax, fig


if __name__ == '__main__':
    new_map = MapPlotter()
    new_map.final_map()
    new_map.path_planning_sim()
