import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import math

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

        ax.set_xlim(0, self.map_width)
        ax.set_ylim(0, self.map_height)
        ax.invert_yaxis()

        ax.quiver(robot[0] + 0.5, robot[1] + 0.5, direction[0], direction[1], color='r')
        # ax.quiver(robot[0] + 0.5, robot[1] + 0.5, math.cos(math.radians(0)), math.sin(math.radians(0)), color='g')
        ax.scatter(robot[0] + 0.5, robot[1] + 0.5, marker='P')

        majorx_ticks = np.arange(0, self.map_width-1, 10)
        minorx_ticks = np.arange(0, self.map_width-1, 1)
        majory_ticks = np.arange(0, self.map_height-1, 10)
        minory_ticks = np.arange(0, self.map_height-1, 1)

        x_lim = (self.map_width-1) * self.cell_size/2
        y_lim = (self.map_height-1) * self.cell_size/2
        majorx_labels = np.arange(-x_lim, x_lim, 10 * self.cell_size)
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

    # def obstacle_indexes2(self, robot, lidar):
    #     obs_indx = []
    #     robot_x, robot_y, robot_ang = robot
    #     for i in range(len(lidar)):
    #         if lidar[i] > 5 or lidar[i] < 0.5:
    #             continue
    #         ang = math.radians(240) / len(lidar) * i + robot_ang - math.radians(120)
    #         deb = lidar[i]
    #         obs_pos = robot_x + lidar[i] * math.cos(ang), robot_y - lidar[i] * math.sin(ang)
    #         obs_cell = self.pos_to_cell(*obs_pos)
    #         obs_indx.append([obs_cell[0], obs_cell[1]])
    #     obs_indx = np.array(obs_indx)
    #     obs_indx = np.array(np.split(obs_indx, 2, axis=1))
    #     obs_indx = obs_indx.reshape(obs_indx.shape[0], obs_indx.shape[1])
    #     return obs_indx

    def map_in_time(self, index):
        cur_data = self.log_data[index]
        robot_x, robot_y, robot_ang = cur_data[0]
        print(cur_data[0])

        robot_indx = self.pos_to_cell(robot_x, robot_y)
        print(robot_indx)
        print(math.degrees(robot_ang))

        robot_dir = math.cos(robot_ang), math.sin(robot_ang)

        obstacle_indx = self.obstacle_indexes(robot_indx, robot_ang, cur_data[1])
        self.map[obstacle_indx[1], obstacle_indx[0]] = 1
        self.print_map(robot_indx, robot_dir)

if __name__ == '__main__':
    new_map = MapPlotter()
    new_map.map_in_time(0)