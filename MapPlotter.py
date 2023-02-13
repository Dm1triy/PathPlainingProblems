import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import math

class MapPlotter:
    def __init__(self, log='examp4.txt'):
        self.cell_size = 0.05  # m
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

    def print_map(self, robot):
        fig, ax = plt.subplots(figsize=(15, 8))
        cmap = colors.ListedColormap(['White', 'Blue'])
        ax.pcolor(self.map, cmap=cmap, snap=True)

        ax.set_xlim(0, self.map_width)
        ax.set_ylim(0, self.map_height)

        ax.scatter(robot[0] + 0.5, robot[1] + 0.5, marker='P')

        majorx_ticks = np.arange(0, self.map_width-1, 10)
        minorx_ticks = np.arange(0, self.map_width-1, 1)
        majory_ticks = np.arange(0, self.map_height-1, 10)
        minory_ticks = np.arange(0, self.map_height-1, 1)

        x_lim = (self.map_width-1) * self.cell_size/2
        y_lim = (self.map_height-1) * self.cell_size/2
        majorx_labels = np.arange(-x_lim, x_lim, 10 * self.cell_size)
        majory_labels = np.arange(-y_lim, y_lim, 10 * self.cell_size)

        # ax.set_xticks(majorx_ticks, labels=majorx_labels)
        # ax.set_xticks(minorx_ticks, labels=[], minor=True)
        # ax.set_yticks(majory_ticks, labels=majory_labels)
        # ax.set_yticks(minory_ticks, labels=[], minor=True)

        ax.set_xticks(majorx_ticks)
        ax.set_xticks(minorx_ticks, minor=True)
        ax.set_yticks(majory_ticks)
        ax.set_yticks(minory_ticks, minor=True)

        ax.grid(which='Both')

        fig.tight_layout()
        plt.show()

    def pos_to_cell(self, x, y):
        return int(self.start_x + x/self.cell_size), int(self.start_y - y/self.cell_size)

    def cell_to_pos(self, x, y):
        # x positive if index is higher (right); y positive if index is smaller (up)
        return np.round((x - self.start_x) * self.cell_size, 2), \
               np.round((self.start_y - y) * self.cell_size, 2)

    def map_in_time(self, index):
        cur_data = self.log_data[index]
        robot_x, robot_y, robot_ang = cur_data[0]
        robot_indx = self.pos_to_cell(robot_x, robot_y)




        self.print_map(robot_indx)

if __name__ == '__main__':
    new_map = MapPlotter()
    new_map.map_in_time(0)