# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Taichi implementation by Ye Kuang (k-ye)
# Improved by Yiwei Xiang (NextoneX)

import math
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)

screen_res = (800, 600)
screen_to_world_ratio = 10.0
boundary = (screen_res[0] / screen_to_world_ratio,
            screen_res[1] / screen_to_world_ratio)
cell_size = 2.51
cell_recpr = 1.0 / cell_size
mouse = (0.0, 0.0)

def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s

grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1))

# Color
bg_color = 0xf6f6ee
particle_color = 0x99d8f5
boundary_color = 0xebaca2

max_num_particles = 8192
max_num_particles_per_cell = 100
max_num_neighbors = 100
time_delta = 1.0 / 20.0
epsilon = 1e-5
particle_radius = 3.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

# PBF params
h_ = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 5
corr_deltaQ_coeff = 0.3
corrK = 0.001
# Need ti.pow()
# corrN = 4.0
neighbor_radius = h_ * 1.05

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

gravity = ti.Vector.field(2, dtype=float, shape=())
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())
period = ti.field(dtype=ti.i32, shape=())
range__ = ti.field(dtype=float, shape=())
period_ = ti.field(dtype=ti.i32, shape=())
range_ = ti.field(dtype=float, shape=())

old_positions = ti.Vector.field(2, float)
positions = ti.Vector.field(2, float)
velocities = ti.Vector.field(2, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
lambdas = ti.field(float)
position_deltas = ti.Vector.field(2, float)
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float)

num_particles = ti.field(dtype=ti.i32, shape=())
ti.root.dense(ti.i, max_num_particles).place(old_positions, positions, velocities)

grid_snode = ti.root.dense(ti.ij, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.k, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, max_num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
ti.root.dense(ti.i, max_num_particles).place(lambdas, position_deltas)
ti.root.place(board_states)
gravity[None] = [0, -1]
period_[None] = 90
range_[None] = 16.0

@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result

@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result

@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), h_) / poly6_value(corr_deltaQ_coeff * h_,
                                                     h_)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x

@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)

@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[
        1] < grid_size[1]

@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1]
                      ]) - particle_radius_in_world
    for i in ti.static(range(2)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p

@ti.kernel
def new_particle(new_num: ti.i32, pos_x: ti.f32, pos_y: ti.f32):
    # Taichi doesn't support using vectors as kernel arguments yet, so we pass scalars
    n = num_particles[None]
    m = ti.min(n+new_num, max_num_particles)
    for i in range(n, m):
        positions[i] = ti.Vector([(pos_x + (i - n) * ti.random() * 0.01)* boundary[0], 
                                  (pos_y + (i - n) * ti.random() * 0.01) * boundary[1] ])
        velocities[i] = ti.Vector([0, 0])
    num_particles[None] += new_num

@ti.kernel
def move_board():
    # probably more accurate to exert force on particles according to hooke's law.
    b = board_states[None]
    b[1] += 1.0
    if b[1] >= 2 * period_[None]:
        b[1] = 0
        period_[None] = period[None]
        range_[None] = range__[None]
    b[0] += -ti.sin(b[1] * np.pi / period_[None]) * range_[None] * time_delta * 50 / period_[None]
    board_states[None] = b


@ti.kernel
def prologue(attraction: ti.f32):
    n = num_particles[None]
    # save old positions
    for i in range(n):
        old_positions[i] = positions[i]
    # apply gravity within boundary
    for i in range(n):
        g = gravity[None]
        pos, vel = positions[i], velocities[i]
        dist = attractor_pos[None] * boundary - pos
        vel += g * time_delta * 9.8
        vel += dist / (0.01 + dist.norm()) * attractor_strength[None] * time_delta * attraction
        pos += vel * time_delta
        positions[i] = confine_position_to_boundary(pos)

    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1

    # update grid
    for p_i in range(n):
        cell = get_cell(positions[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # find particle neighbors
    for p_i in range(n):
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (
                            pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i


@ti.kernel
def substep():
    n = num_particles[None]
    # compute lambdas
    # Eq (8) ~ (11)
    for p_i in range(n):
        pos_i = positions[p_i]

        grad_i = ti.Vector([0.0, 0.0])
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            grad_j = spiky_gradient(pos_ji, h_)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            # Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), h_)

        # Eq(1)
        density_constraint = (mass * density_constraint / rho0) - 1.0

        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr +
                                                lambda_epsilon)
    # compute position deltas
    # Eq(12), (14)
    for p_i in range(n):
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        pos_delta_i = ti.Vector([0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            lambda_j = lambdas[p_j]
            pos_ji = pos_i - positions[p_j]
            scorr_ij = compute_scorr(pos_ji)
            pos_delta_i += (lambda_i + lambda_j + scorr_ij) * \
                spiky_gradient(pos_ji, h_)

        pos_delta_i /= rho0
        position_deltas[p_i] = pos_delta_i
    # apply position deltas
    for i in range(n):
        positions[i] += position_deltas[i]


@ti.kernel
def epilogue():
    n = num_particles[None]
    # confine to boundary
    for i in range(n):
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    # update velocities
    for i in range(n):
        velocities[i] = (positions[i] - old_positions[i]) / time_delta
    # no vorticity/xsph because we cannot do cross product in 2D...


def run_pbf(attraction):
    prologue(attraction)
    for _ in range(pbf_num_iters):
        substep()
    epilogue()


def render(gui):
    n = num_particles[None]
    gui.clear(bg_color)
    pos_np = positions.to_numpy()
    for j in range(2):
        pos_np[:, j] *= screen_to_world_ratio / screen_res[j]
    for i in range(n):
        gui.circle(pos=pos_np[i], radius=particle_radius, color=0x6c79db)
    gui.rect((0, 0), (board_states[None][0] / boundary[0], 1),
             radius=1.5,
             color=boundary_color)
    mouse = gui.get_cursor_pos()
    gui.circle((mouse[0], mouse[1]), color=0x996633, radius=(5 + attractor_strength[None] * 10))
    gui.text(content=
            'Hint:',
            pos=(0, 0.99),color=0x0)
    gui.text(content=
            'Left Mouse Button: create particle;  Right Mouse Button: generate attraction',
            pos=(0, 0.95),color=0x0)
    gui.text(content='R: restart;  Space: pause_board;  WSAD/arrow keys: control gravity',
            pos=(0, 0.91),color=0x0)
    # gui.text(content=f'Y: Spring Young\'s modulus {spring_Y[None]:.1f}',
    #         pos=(0, 0.9),color=0x0)
    # gui.text(content=f'D: Drag damping {drag_damping[None]:.2f}',
    #         pos=(0, 0.85),color=0x0)
    # gui.text(content=f'X: Dashpot damping {dashpot_damping[None]:.2f}',
    #         pos=(0, 0.8),color=0x0)
    gui.show()


@ti.kernel
def init_particles(initial: ti.i32):
    num_particles[None] = initial
    for i in range(num_particles[None]):
        delta = h_ * 0.8
        offs = ti.Vector([(boundary[0] - delta * (num_particles[None] // 20)) * 0.5,
                          boundary[1] * 0.02])
        positions[i] = ti.Vector([i % (num_particles[None] // 20), i // (num_particles[None] // 20)
                                  ]) * delta + offs
        for c in ti.static(range(2)):
            velocities[i][c] = (ti.random() - 0.5) * 4
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])


# def print_stats():
#     print('PBF stats:')
#     num = grid_num_particles.to_numpy()
#     avg, max_ = np.mean(num), np.max(num)
#     print(f'  #particles per cell: avg={avg:.2f} max={max_}')
#     num = particle_num_neighbors.to_numpy()
#     avg, max_ = np.mean(num), np.max(num)
#     print(f'  #neighbors per particle: avg={avg:.2f} max={max_}')

def main():
    gui = ti.GUI('PBF2D', screen_res)
    Initial_particles = gui.slider('Initial Particles', 
                                   0, 4000, step=50)
    Initial_particles.value = 1600
    new_particles = gui.slider('New Particles', 
                                   1, 10, step=1)
    new_particles.value = 1
    attraction = gui.slider('Attraction Strength', 
                                   0, 30, step=1)
    attraction.value = 15
    board_period = gui.slider('Board Period', 
                                   60, 180, step=1)
    board_period.value = 90
    borad_range = gui.slider('Board Range', 
                                   0, 32, step=1)
    borad_range.value = 16
    pause = gui.button('Pause_board')
    p_num = gui.label('Particle_num')
    max_num = gui.label('Particle_max')
    max_num.value = 4096
    init_particles(int(Initial_particles.value))
    print(f'boundary={boundary} grid={grid_size} cell_size={cell_size}')
    paused = False
    while gui.running:
        p_num.value = num_particles[None]
        range__[None] = borad_range.value
        period[None] = int(board_period.value)
        mouse = gui.get_cursor_pos()
        attractor_pos[None] = [mouse[0], mouse[1]]
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key in [ti.GUI.SPACE, 'p', pause]:
                paused = not paused
            elif e.key == 'r':
                init_particles(int(Initial_particles.value))
                paused = False
        attractor_strength[None] = 0
        if gui.is_pressed(ti.GUI.LMB):
            # print(f'  #num_particles={num_particles[None]}')
            new_particle(int(new_particles.value), mouse[0], mouse[1])
        if gui.is_pressed(ti.GUI.RMB):
            attractor_strength[None] = 1 
        if gui.is_pressed(ti.GUI.LEFT, 'a'):
            gravity[None] = [-1, 0]
        if gui.is_pressed(ti.GUI.RIGHT, 'd'):
            gravity[None] = [1, 0]
        if gui.is_pressed(ti.GUI.UP, 'w'):
            gravity[None] = [0, 1]
        if gui.is_pressed(ti.GUI.DOWN, 's'):
            gravity[None] = [0, -1]
        if not paused:
            move_board()
        run_pbf(attraction.value)
        # if gui.frame % 20 == 1:
        #     print_stats()
        render(gui)


if __name__ == '__main__':
    main()
