import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import math
from rdp import rdp
class MapPlotter:
    def __init__(self, log='examp4.txt'):
        self.cell_size = 0.1  # m
        self.map_width = 251
        self.map_height = 251
        self.map = np.array([[0] * self.map_width]*self.map_height)
        self.start_x = self.map_width//2
        self.start_y = self.map_height//2
        self.log_data = self.read_log(log)

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

    def print_map(self, robot, direction):
        fig, ax = plt.subplots(figsize=(15, 8))
        cmap = colors.ListedColormap(['White', 'Blue'])
        ax.pcolor(self.map, cmap=cmap, snap=True)

        ax.set_xlim(self.start_x, self.map_width)  # ax.set_xlim(0, self.map_width)
        ax.set_ylim(0, self.map_height)
        ax.invert_yaxis()

        ax.quiver(robot[0] + 0.5, robot[1] + 0.5, direction[0], direction[1], color='r')
        ax.scatter(robot[0] + 0.5, robot[1] + 0.5, marker='P')

        majorx_ticks = np.arange(self.start_x, self.map_width-1, 10) # majorx_ticks = np.arange(0, self.map_width-1, 10)
        minorx_ticks = np.arange(self.start_x, self.map_width-1, 1) # minorx_ticks = np.arange(0, self.map_width-1, 1)
        majory_ticks = np.arange(0, self.map_height-1, 10)
        minory_ticks = np.arange(0, self.map_height-1, 1)

        x_lim = (self.map_width-1) * self.cell_size/2
        y_lim = (self.map_height-1) * self.cell_size/2
        majorx_labels = np.arange(0, x_lim, 10 * self.cell_size)    #majorx_labels = np.arange(-x_lim, x_lim, 10 * self.cell_size)

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
            ang = math.radians(240)/len(lidar) * i + robot_ang - math.radians(120)
            # lidar radius in meter therefore radius*cos(ang) should be converted to the index
            local_indx = int(lidar[i] * math.cos(ang)/self.cell_size),\
                         int(lidar[i] * math.sin(ang)/self.cell_size)
            obs_indx.append([robot_pos[0] + local_indx[0], robot_pos[1] - local_indx[1]])
        # convert [[i1, j1], [i2, j2], ...] into [[i1, i2, ...], [j1, j1, ...]]
        obs_indx = np.array(obs_indx)
        obs_indx = np.array(np.split(obs_indx, 2, axis=1))
        obs_indx = obs_indx.reshape(obs_indx.shape[0], obs_indx.shape[1])
        return obs_indx

    def map_in_time(self, index):
        cur_data = self.log_data[index]
        robot_x, robot_y, robot_ang = cur_data[0]

        robot_indx = self.pos_to_cell(robot_x, robot_y)
        robot_dir = math.cos(robot_ang), math.sin(robot_ang)

        obstacle_indx = self.obstacle_indexes(robot_indx, robot_ang, cur_data[1])
        self.map[obstacle_indx[1], obstacle_indx[0]] = 1
        # self.print_map(robot_indx, robot_dir)
        return robot_indx, robot_dir

    def reduce_map(self):
        obs_indexes = np.where(self.map == 1)
        self.map[obs_indexes[0], obs_indexes[1]] = 0
        obs_indexes = np.stack(obs_indexes, axis=1)

        new_indexes = rdp(obs_indexes, epsilon=1)
        new_indexes = np.array(np.split(new_indexes, 2, axis=1))
        new_indexes = new_indexes.reshape(new_indexes.shape[0], new_indexes.shape[1])
        self.map[new_indexes[0], new_indexes[1]] = 1

    def final_map(self):
        robot_pos, robot_dir = [], []
        for i in range(len(self.log_data)):
            if i%5 == 0:
                self.reduce_map()
            robot_pos, robot_dir = self.map_in_time(i)
        self.reduce_map()
        self.print_map(robot_pos, robot_dir)
        self.rectangle_map()
        self.print_map(robot_pos, robot_dir)

    def rectangle_map(self):
        indexes = np.where(self.map == 1)
        self.map[indexes[0], indexes[1]] = 0
        min_y, max_y = int(np.quantile(indexes[0], .1)), int(np.quantile(indexes[0], .9))
        min_x, max_x = int(np.quantile(indexes[1], .1)), int(np.quantile(indexes[1], .8))
        vertical = np.arange(min_y, max_y, 1)
        horizontal = np.arange(min_x, max_x, 1)
        self.map[vertical, min_x] = 1
        self.map[vertical, max_x] = 1
        self.map[min_y, horizontal] = 1
        self.map[min_y-1, horizontal] = 1
        self.map[max_y, horizontal] = 1
        self.map[max_y+1, horizontal] = 1


if __name__ == '__main__':
    new_map = MapPlotter()
    new_map.final_map()
