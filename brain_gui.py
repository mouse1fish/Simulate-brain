import tkinter as tk
from tkinter import ttk
import numpy as np
import math
from brain.brain import Brain


COLORS = {
    'bg_dark': '#0a0a1a',
    'bg_panel': '#111128',
    'bg_card': '#1a1a3e',
    'border': '#2a2a5e',
    'text': '#e0e0ff',
    'text_dim': '#7878a8',
    'accent': '#4a9eff',
    'accent2': '#ff6b6b',
    'green': '#4aff8b',
    'yellow': '#ffdd4a',
    'orange': '#ff8c4a',
    'purple': '#b44aff',
    'cyan': '#4affee',
}

REGION_COLORS = {
    'visual': '#4a9eff',
    'auditory': '#ff6b6b',
    'motor': '#4aff8b',
    'association': '#b44aff',
    'memory': '#ffdd4a',
}

REGION_LABELS_CN = {
    'visual': '视觉区',
    'auditory': '听觉区',
    'motor': '运动区',
    'association': '联合区',
    'memory': '记忆区',
}

REGION_3D_CENTER = {
    'visual': np.array([0.0, 0.0, -0.7]),
    'auditory': np.array([-0.7, 0.0, 0.0]),
    'motor': np.array([0.7, 0.0, 0.0]),
    'association': np.array([0.0, 0.7, 0.0]),
    'memory': np.array([0.0, 0.0, 0.7]),
}

REGION_3D_RADIUS = {
    'visual': 0.45,
    'auditory': 0.40,
    'motor': 0.40,
    'association': 0.50,
    'memory': 0.45,
}


def potential_to_color(potential, threshold=1.0):
    ratio = min(potential / threshold, 1.0) if threshold > 0 else 0
    if ratio < 0.05:
        return '#0a0a2a'
    elif ratio < 0.2:
        r = int(10 + ratio * 5 * 30)
        g = int(10 + ratio * 5 * 60)
        b = int(40 + ratio * 5 * 80)
        return f'#{r:02x}{g:02x}{b:02x}'
    elif ratio < 0.5:
        t = (ratio - 0.2) / 0.3
        r = int(40 + t * 60)
        g = int(70 + t * 130)
        b = int(120 + t * 80)
        return f'#{r:02x}{g:02x}{b:02x}'
    elif ratio < 0.8:
        t = (ratio - 0.5) / 0.3
        r = int(100 + t * 155)
        g = int(200 - t * 60)
        b = int(200 - t * 150)
        return f'#{r:02x}{g:02x}{b:02x}'
    else:
        t = min((ratio - 0.8) / 0.2, 1.0)
        r = 255
        g = int(140 + t * 115)
        b = int(50 + t * 205)
        return f'#{r:02x}{g:02x}{b:02x}'


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(r, g, b):
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return f'#{r:02x}{g:02x}{b:02x}'


def blend_color(hex_color, factor):
    r, g, b = hex_to_rgb(hex_color)
    return rgb_to_hex(r * factor, g * factor, b * factor)


class Renderer3D:
    def __init__(self, canvas, brain):
        self.canvas = canvas
        self.brain = brain
        self.rot_x = 0.3
        self.rot_y = -0.5
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.drag_start = None
        self.drag_btn = None
        self.neuron_positions = None
        self.neuron_items = []
        self.connection_items = []
        self.selected_neuron_3d = None
        self.show_connections = False
        self._generate_positions()

        canvas.bind('<ButtonPress-1>', self._on_drag_start)
        canvas.bind('<B1-Motion>', self._on_drag_move)
        canvas.bind('<ButtonPress-3>', self._on_rdrag_start)
        canvas.bind('<B3-Motion>', self._on_rdrag_move)
        canvas.bind('<MouseWheel>', self._on_scroll)
        canvas.bind('<Button-4>', self._on_scroll_linux)
        canvas.bind('<Button-5>', self._on_scroll_linux)
        canvas.bind('<ButtonPress-2>', self._on_click_3d)

    def _generate_positions(self):
        rng = np.random.default_rng(123)
        n = self.brain.num_neurons
        positions = np.zeros((n, 3))

        for name, (start, end) in Brain.REGIONS.items():
            center = REGION_3D_CENTER[name]
            radius = REGION_3D_RADIUS[name]
            count = end - start
            for i in range(count):
                while True:
                    offset = rng.uniform(-radius, radius, 3)
                    if np.linalg.norm(offset) <= radius:
                        break
                positions[start + i] = center + offset

        self.neuron_positions = positions

    def _rotation_matrix(self):
        cx, sx = math.cos(self.rot_x), math.sin(self.rot_x)
        cy, sy = math.cos(self.rot_y), math.sin(self.rot_y)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        return Ry @ Rx

    def project(self, points_3d):
        R = self._rotation_matrix()
        rotated = points_3d @ R.T

        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600
        fov = 3.0 * self.zoom
        cx_screen = cw / 2 + self.pan_x
        cy_screen = ch / 2 + self.pan_y

        z = rotated[:, 2]
        depth_factor = fov / (fov + z + 2.5)
        depth_factor = np.clip(depth_factor, 0.1, 5.0)

        screen_x = rotated[:, 0] * depth_factor * min(cw, ch) * 0.35 + cx_screen
        screen_y = -rotated[:, 1] * depth_factor * min(cw, ch) * 0.35 + cy_screen

        return screen_x, screen_y, z, depth_factor

    def _on_drag_start(self, event):
        self.drag_start = (event.x, event.y)
        self.drag_btn = 1

    def _on_rdrag_start(self, event):
        self.drag_start = (event.x, event.y)
        self.drag_btn = 3

    def _on_drag_move(self, event):
        if self.drag_start is None:
            return
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        self.rot_y += dx * 0.005
        self.rot_x += dy * 0.005
        self.rot_x = max(-math.pi / 2, min(math.pi / 2, self.rot_x))
        self.drag_start = (event.x, event.y)

    def _on_rdrag_move(self, event):
        if self.drag_start is None:
            return
        dx = event.x - self.drag_start[0]
        dy = event.y - self.drag_start[1]
        self.pan_x += dx
        self.pan_y += dy
        self.drag_start = (event.x, event.y)

    def _on_scroll(self, event):
        if event.delta > 0:
            self.zoom *= 1.1
        else:
            self.zoom /= 1.1
        self.zoom = max(0.3, min(5.0, self.zoom))

    def _on_scroll_linux(self, event):
        if event.num == 4:
            self.zoom *= 1.1
        elif event.num == 5:
            self.zoom /= 1.1
        self.zoom = max(0.3, min(5.0, self.zoom))

    def _on_click_3d(self, event):
        sx, sy, z, df = self.project(self.neuron_positions)
        dists = np.sqrt((sx - event.x) ** 2 + (sy - event.y) ** 2)
        closest = np.argmin(dists)
        if dists[closest] < 15:
            self.selected_neuron_3d = closest
        else:
            self.selected_neuron_3d = None

    def render(self):
        self.canvas.delete('all')
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600

        self._draw_axes(cw, ch)
        self._draw_region_labels(cw, ch)

        sx, sy, z, df = self.project(self.neuron_positions)

        order = np.argsort(-z)

        potentials = self.brain.potentials
        fired = self.brain.fired
        is_exc = self.brain.is_excitatory
        thresholds = self.brain.thresholds

        for idx in order:
            x = sx[idx]
            y = sy[idx]
            depth = df[idx]

            if x < -20 or x > cw + 20 or y < -20 or y > ch + 20:
                continue

            region = self.brain._get_region(idx)
            region_color = REGION_COLORS.get(region, '#4a9eff')

            if fired[idx]:
                base_color = '#ffee4a' if is_exc[idx] else '#ff4a6b'
                radius = max(2, 5 * depth)
                glow_r = radius + 3
                glow_color = blend_color(base_color, 0.3)
                self.canvas.create_oval(
                    x - glow_r, y - glow_r, x + glow_r, y + glow_r,
                    fill=glow_color, outline='', tags='neuron')
                self.canvas.create_oval(
                    x - radius, y - radius, x + radius, y + radius,
                    fill=base_color, outline='', tags='neuron')
            else:
                ratio = min(potentials[idx] / thresholds[idx], 1.0)
                if ratio < 0.05:
                    brightness = 0.15 + 0.15 * depth
                    color = blend_color(region_color, brightness)
                elif ratio < 0.3:
                    t = ratio / 0.3
                    brightness = 0.3 + 0.4 * t
                    color = blend_color(region_color, brightness)
                elif ratio < 0.7:
                    t = (ratio - 0.3) / 0.4
                    r1, g1, b1 = hex_to_rgb(region_color)
                    r2, g2, b2 = hex_to_rgb('#ffffff')
                    r = int(r1 + (r2 - r1) * t * 0.5)
                    g = int(g1 + (g2 - g1) * t * 0.5)
                    b = int(b1 + (b2 - b1) * t * 0.5)
                    color = rgb_to_hex(r, g, b)
                else:
                    t = (ratio - 0.7) / 0.3
                    color = blend_color('#ffdd4a', 0.8 + 0.2 * t)

                radius = max(1.5, (2.5 + ratio * 2) * depth)
                self.canvas.create_oval(
                    x - radius, y - radius, x + radius, y + radius,
                    fill=color, outline='', tags='neuron')

        if self.selected_neuron_3d is not None:
            nid = self.selected_neuron_3d
            nx, ny = sx[nid], sy[nid]
            r = max(4, 8 * df[nid])
            self.canvas.create_oval(
                nx - r, ny - r, nx + r, ny + r,
                fill='', outline='#ffffff', width=2, tags='selected')

        self.canvas.create_text(10, ch - 10,
                                text="左键拖拽:旋转 | 右键拖拽:平移 | 滚轮:缩放 | 中键:选择神经元",
                                fill=COLORS['text_dim'], anchor='sw',
                                font=('Microsoft YaHei UI', 8))

    def _draw_axes(self, cw, ch):
        origin = np.array([[0, 0, 0]])
        axes = np.array([[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]])
        labels = ['X', 'Y', 'Z']
        axis_colors = ['#ff4444', '#44ff44', '#4444ff']

        ox, oy, _, _ = self.project(origin)
        ax, ay, _, _ = self.project(axes)

        for i in range(3):
            self.canvas.create_line(
                ox[0], oy[0], ax[i], ay[i],
                fill=axis_colors[i], width=1, dash=(3, 3))
            self.canvas.create_text(
                ax[i] + 5, ay[i] - 5, text=labels[i],
                fill=axis_colors[i], font=('Consolas', 8))

    def _draw_region_labels(self, cw, ch):
        for name, center in REGION_3D_CENTER.items():
            label_pos = np.array([center * 1.3])
            sx, sy, z, df = self.project(label_pos)
            color = REGION_COLORS[name]
            label = REGION_LABELS_CN[name]
            self.canvas.create_text(
                sx[0], sy[0], text=label, fill=color,
                font=('Microsoft YaHei UI', 9, 'bold'), tags='label')


class BrainSimulatorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Brain Simulator - 大脑模拟器 3D")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 750)
        self.root.configure(bg=COLORS['bg_dark'])

        self.brain = Brain(num_neurons=1000, avg_connections=1000, seed=42)
        self.running = False
        self.speed = 50
        self.tick_count = 0
        self.selected_neuron = None
        self.activity_history = []
        self.region_history = {name: [] for name in Brain.REGIONS}
        self.max_history = 200
        self.view_mode = '3d'

        self._setup_styles()
        self._build_ui()
        self._update_display()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background=COLORS['bg_dark'])
        style.configure('Panel.TFrame', background=COLORS['bg_panel'])

    def _build_ui(self):
        self.root.columnconfigure(0, weight=4)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = tk.Frame(self.root, bg=COLORS['bg_dark'], height=48)
        header.grid(row=0, column=0, columnspan=2, sticky='ew')
        header.grid_propagate(False)

        tk.Label(header, text="🧠 Brain Simulator 3D - 大脑模拟器",
                 bg=COLORS['bg_dark'], fg=COLORS['accent'],
                 font=('Microsoft YaHei UI', 15, 'bold')).pack(side='left', padx=15, pady=8)

        self.header_status = tk.Label(header, text="就绪 | 1000 神经元",
                                      bg=COLORS['bg_dark'], fg=COLORS['text_dim'],
                                      font=('Consolas', 10))
        self.header_status.pack(side='right', padx=15, pady=8)

        viz_frame = tk.Frame(self.root, bg=COLORS['bg_dark'])
        viz_frame.grid(row=1, column=0, sticky='nsew', padx=(8, 4), pady=4)
        viz_frame.rowconfigure(1, weight=5)
        viz_frame.rowconfigure(2, weight=2)
        viz_frame.columnconfigure(0, weight=1)

        view_switch = tk.Frame(viz_frame, bg=COLORS['bg_dark'])
        view_switch.grid(row=0, column=0, sticky='ew', pady=(0, 4))

        self.view_3d_btn = tk.Button(view_switch, text="🌐 3D视图",
                                     bg=COLORS['accent'], fg='#ffffff',
                                     font=('Microsoft YaHei UI', 10, 'bold'),
                                     relief='flat', command=lambda: self._switch_view('3d'),
                                     cursor='hand2')
        self.view_3d_btn.pack(side='left', padx=2)

        self.view_2d_btn = tk.Button(view_switch, text="📊 2D矩阵",
                                     bg='#1a1a3e', fg=COLORS['text_dim'],
                                     font=('Microsoft YaHei UI', 10, 'bold'),
                                     relief='flat', command=lambda: self._switch_view('2d'),
                                     cursor='hand2')
        self.view_2d_btn.pack(side='left', padx=2)

        main_card = tk.Frame(viz_frame, bg='#050510',
                             highlightbackground=COLORS['border'], highlightthickness=1)
        main_card.grid(row=1, column=0, sticky='nsew')
        main_card.rowconfigure(0, weight=1)
        main_card.columnconfigure(0, weight=1)

        self.canvas_3d = tk.Canvas(main_card, bg='#050510', highlightthickness=0)
        self.canvas_3d.grid(row=0, column=0, sticky='nsew')

        self.canvas_2d = tk.Canvas(main_card, bg='#050510', highlightthickness=0)
        self.neuron_rects = []
        self.neuron_grid_cols = 40
        self.neuron_grid_rows = 25

        self.renderer = Renderer3D(self.canvas_3d, self.brain)

        self.canvas_2d_visible = False
        self.canvas_3d_visible = True

        bottom_frame = tk.Frame(viz_frame, bg=COLORS['bg_dark'])
        bottom_frame.grid(row=2, column=0, sticky='nsew')
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)
        bottom_frame.rowconfigure(0, weight=1)

        region_card = tk.Frame(bottom_frame, bg=COLORS['bg_card'],
                               highlightbackground=COLORS['border'], highlightthickness=1)
        region_card.grid(row=0, column=0, sticky='nsew', padx=(0, 2))
        region_card.rowconfigure(1, weight=1)
        region_card.columnconfigure(0, weight=1)

        tk.Label(region_card, text="  脑区活动",
                 bg=COLORS['bg_card'], fg=COLORS['accent'],
                 font=('Microsoft YaHei UI', 10, 'bold'), anchor='w').grid(
            row=0, column=0, sticky='ew', padx=8, pady=(6, 2))

        self.region_canvas = tk.Canvas(region_card, bg='#050510', highlightthickness=0)
        self.region_canvas.grid(row=1, column=0, sticky='nsew', padx=6, pady=(2, 6))

        graph_card = tk.Frame(bottom_frame, bg=COLORS['bg_card'],
                              highlightbackground=COLORS['border'], highlightthickness=1)
        graph_card.grid(row=0, column=1, sticky='nsew', padx=(2, 0))
        graph_card.rowconfigure(1, weight=1)
        graph_card.columnconfigure(0, weight=1)

        tk.Label(graph_card, text="  活动历史曲线",
                 bg=COLORS['bg_card'], fg=COLORS['accent'],
                 font=('Microsoft YaHei UI', 10, 'bold'), anchor='w').grid(
            row=0, column=0, sticky='ew', padx=8, pady=(6, 2))

        self.graph_canvas = tk.Canvas(graph_card, bg='#050510', highlightthickness=0)
        self.graph_canvas.grid(row=1, column=0, sticky='nsew', padx=6, pady=(2, 6))

        right_panel = tk.Frame(self.root, bg=COLORS['bg_panel'],
                               highlightbackground=COLORS['border'], highlightthickness=1,
                               width=280)
        right_panel.grid(row=1, column=1, sticky='nsew', padx=(4, 8), pady=4)
        right_panel.grid_propagate(False)
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)

        ctrl_frame = tk.Frame(right_panel, bg=COLORS['bg_panel'])
        ctrl_frame.grid(row=0, column=0, sticky='ew', padx=6, pady=6)
        ctrl_frame.columnconfigure(0, weight=1)

        btn_row1 = tk.Frame(ctrl_frame, bg=COLORS['bg_panel'])
        btn_row1.grid(row=0, column=0, sticky='ew', pady=2)
        btn_row1.columnconfigure(0, weight=1)
        btn_row1.columnconfigure(1, weight=1)

        self.start_btn = tk.Button(btn_row1, text="▶ 运行", bg='#1a4a2a', fg='#4aff8b',
                                   font=('Microsoft YaHei UI', 10, 'bold'), relief='flat',
                                   command=self._toggle_run, cursor='hand2')
        self.start_btn.grid(row=0, column=0, sticky='ew', padx=(0, 2))

        self.step_btn = tk.Button(btn_row1, text="⏭ 单步", bg='#1a2a4a', fg='#4a9eff',
                                  font=('Microsoft YaHei UI', 10, 'bold'), relief='flat',
                                  command=self._step_once, cursor='hand2')
        self.step_btn.grid(row=0, column=1, sticky='ew', padx=(2, 0))

        btn_row2 = tk.Frame(ctrl_frame, bg=COLORS['bg_panel'])
        btn_row2.grid(row=1, column=0, sticky='ew', pady=2)
        btn_row2.columnconfigure(0, weight=1)

        self.reset_btn = tk.Button(btn_row2, text="🔄 重置大脑", bg='#4a1a1a', fg='#ff6b6b',
                                   font=('Microsoft YaHei UI', 10, 'bold'), relief='flat',
                                   command=self._reset, cursor='hand2')
        self.reset_btn.grid(row=0, column=0, sticky='ew')

        speed_frame = tk.Frame(ctrl_frame, bg=COLORS['bg_panel'])
        speed_frame.grid(row=2, column=0, sticky='ew', pady=(6, 2))
        speed_frame.columnconfigure(1, weight=1)

        tk.Label(speed_frame, text="速度:", bg=COLORS['bg_panel'], fg=COLORS['text'],
                 font=('Microsoft YaHei UI', 9)).grid(row=0, column=0, padx=(0, 4))

        self.speed_var = tk.IntVar(value=50)
        speed_scale = tk.Scale(speed_frame, from_=1, to=200, orient='horizontal',
                               variable=self.speed_var, bg=COLORS['bg_panel'],
                               fg=COLORS['text'], troughcolor=COLORS['bg_card'],
                               highlightthickness=0, sliderrelief='flat',
                               command=self._on_speed_change)
        speed_scale.grid(row=0, column=1, sticky='ew')

        self.learn_var = tk.BooleanVar(value=True)
        learn_cb = tk.Checkbutton(ctrl_frame, text="🧪 Hebbian学习", variable=self.learn_var,
                                  bg=COLORS['bg_panel'], fg=COLORS['text'],
                                  selectcolor=COLORS['bg_card'],
                                  font=('Microsoft YaHei UI', 9),
                                  command=self._on_learn_toggle)
        learn_cb.grid(row=3, column=0, sticky='w', pady=4)

        stim_frame = tk.LabelFrame(right_panel, text="  ⚡ 刺激控制  ",
                                   bg=COLORS['bg_panel'], fg=COLORS['accent'],
                                   font=('Microsoft YaHei UI', 9, 'bold'))
        stim_frame.grid(row=1, column=0, sticky='nsew', padx=6, pady=4)
        stim_frame.columnconfigure(0, weight=1)

        tk.Label(stim_frame, text="刺激强度:", bg=COLORS['bg_panel'], fg=COLORS['text'],
                 font=('Microsoft YaHei UI', 9)).grid(row=0, column=0, sticky='w', padx=4, pady=(4, 0))
        self.strength_var = tk.DoubleVar(value=3.0)
        strength_scale = tk.Scale(stim_frame, from_=0.5, to=10.0, resolution=0.5,
                                  orient='horizontal', variable=self.strength_var,
                                  bg=COLORS['bg_panel'], fg=COLORS['text'],
                                  troughcolor=COLORS['bg_card'], highlightthickness=0,
                                  sliderrelief='flat')
        strength_scale.grid(row=1, column=0, sticky='ew', padx=4)

        row = 2
        for region_name in ['visual', 'auditory', 'motor', 'association', 'memory']:
            color = REGION_COLORS[region_name]
            label = REGION_LABELS_CN[region_name]
            btn = tk.Button(stim_frame, text=f"🎯 {label}", bg='#1a1a3e', fg=color,
                            font=('Microsoft YaHei UI', 9), relief='flat',
                            command=lambda r=region_name: self._stimulate_region(r),
                            cursor='hand2', activebackground='#2a2a5e')
            btn.grid(row=row, column=0, sticky='ew', padx=4, pady=2)
            row += 1

        pattern_frame = tk.LabelFrame(right_panel, text="  🌊 信号模式  ",
                                      bg=COLORS['bg_panel'], fg=COLORS['accent'],
                                      font=('Microsoft YaHei UI', 9, 'bold'))
        pattern_frame.grid(row=2, column=0, sticky='ew', padx=6, pady=4)
        pattern_frame.columnconfigure(0, weight=1)
        pattern_frame.columnconfigure(1, weight=1)

        patterns_display = [
            ('light', '💡微光'), ('flash', '⚡闪光'),
            ('sound', '🔊声音'), ('noise', '📢噪音'),
            ('move', '🏃运动'), ('kick', '💪强动'),
            ('think', '💭思考'), ('ponder', '🤔沉思'),
            ('remember', '📖记忆'), ('recall', '🔮回忆'),
        ]
        for i, (pname, plabel) in enumerate(patterns_display):
            r, c = divmod(i, 2)
            btn = tk.Button(pattern_frame, text=plabel, bg='#1a1a3e', fg=COLORS['cyan'],
                            font=('Microsoft YaHei UI', 8), relief='flat',
                            command=lambda p=pname: self._inject_pattern(p),
                            cursor='hand2', activebackground='#2a2a5e')
            btn.grid(row=r, column=c, sticky='ew', padx=2, pady=1)

        info_frame = tk.LabelFrame(right_panel, text="  📊 神经元详情  ",
                                   bg=COLORS['bg_panel'], fg=COLORS['accent'],
                                   font=('Microsoft YaHei UI', 9, 'bold'))
        info_frame.grid(row=3, column=0, sticky='ew', padx=6, pady=4)
        info_frame.columnconfigure(0, weight=1)

        self.info_text = tk.Text(info_frame, bg=COLORS['bg_card'], fg=COLORS['text'],
                                 font=('Consolas', 9), height=7, relief='flat',
                                 insertbackground=COLORS['text'], state='disabled',
                                 wrap='word')
        self.info_text.grid(row=0, column=0, sticky='ew', padx=4, pady=4)

        status_frame = tk.Frame(right_panel, bg=COLORS['bg_card'],
                                highlightbackground=COLORS['border'], highlightthickness=1)
        status_frame.grid(row=4, column=0, sticky='ew', padx=6, pady=(4, 6))
        status_frame.columnconfigure(0, weight=1)

        self.status_label = tk.Label(status_frame, text="Tick: 0 | 活跃: 0 | 总放电: 0",
                                     bg=COLORS['bg_card'], fg=COLORS['text_dim'],
                                     font=('Consolas', 9), anchor='w')
        self.status_label.grid(row=0, column=0, sticky='ew', padx=8, pady=6)

    def _switch_view(self, mode):
        self.view_mode = mode
        if mode == '3d':
            self.canvas_3d.grid()
            self.canvas_2d.grid_remove()
            self.view_3d_btn.configure(bg=COLORS['accent'], fg='#ffffff')
            self.view_2d_btn.configure(bg='#1a1a3e', fg=COLORS['text_dim'])
        else:
            self.canvas_3d.grid_remove()
            self.canvas_2d.grid()
            self.view_3d_btn.configure(bg='#1a1a3e', fg=COLORS['text_dim'])
            self.view_2d_btn.configure(bg=COLORS['accent'], fg='#ffffff')
            self._init_neuron_grid()
            self._update_neuron_grid()

    def _init_neuron_grid(self):
        self.canvas_2d.delete('all')
        self.neuron_rects = []
        n = self.brain.num_neurons
        cols = self.neuron_grid_cols
        rows = self.neuron_grid_rows
        cw = self.canvas_2d.winfo_width() or 700
        ch = self.canvas_2d.winfo_height() or 400
        cell_w = cw / cols
        cell_h = ch / rows
        for i in range(n):
            r = i // cols
            c = i % cols
            if r >= rows:
                break
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            rect = self.canvas_2d.create_rectangle(
                x1, y1, x2, y2, fill='#0a0a2a', outline='', width=0)
            self.neuron_rects.append(rect)

    def _toggle_run(self):
        self.running = not self.running
        if self.running:
            self.start_btn.configure(text="⏸ 暂停", bg='#4a3a1a', fg='#ffdd4a')
            self._run_loop()
        else:
            self.start_btn.configure(text="▶ 运行", bg='#1a4a2a', fg='#4aff8b')

    def _step_once(self):
        self.brain.tick()
        self.tick_count += 1
        self._record_history()
        self._update_display()

    def _run_loop(self):
        if not self.running:
            return
        self.brain.tick()
        self.tick_count += 1
        self._record_history()
        self._update_display()
        delay = max(1, 200 - self.speed_var.get() * 2)
        self.root.after(delay, self._run_loop)

    def _on_speed_change(self, val):
        self.speed = int(val)

    def _on_learn_toggle(self):
        self.brain.learning_enabled = self.learn_var.get()

    def _stimulate_region(self, region_name):
        strength = self.strength_var.get()
        self.brain.stimulate_region(region_name, strength)
        if not self.running:
            self._update_display()

    def _inject_pattern(self, pattern_name):
        self.brain.stimulate_pattern(pattern_name)
        if not self.running:
            self._update_display()

    def _reset(self):
        self.running = False
        self.start_btn.configure(text="▶ 运行", bg='#1a4a2a', fg='#4aff8b')
        self.brain.reset()
        self.tick_count = 0
        self.activity_history.clear()
        for k in self.region_history:
            self.region_history[k].clear()
        self._update_display()

    def _record_history(self):
        active_count = int(np.sum(self.brain.fired))
        self.activity_history.append(active_count / self.brain.num_neurons)
        if len(self.activity_history) > self.max_history:
            self.activity_history.pop(0)
        for name, (start, end) in Brain.REGIONS.items():
            region_active = int(np.sum(self.brain.fired[start:end]))
            self.region_history[name].append(region_active / (end - start))
            if len(self.region_history[name]) > self.max_history:
                self.region_history[name].pop(0)

    def _update_display(self):
        if self.view_mode == '3d':
            self.renderer.render()
        else:
            self._update_neuron_grid()

        self._update_region_bars()
        self._update_graph()
        self._update_status()

        nid = None
        if self.view_mode == '3d' and self.renderer.selected_neuron_3d is not None:
            nid = self.renderer.selected_neuron_3d
        elif self.selected_neuron is not None:
            nid = self.selected_neuron
        if nid is not None:
            self._update_neuron_info(nid)

    def _update_neuron_grid(self):
        n = len(self.neuron_rects)
        potentials = self.brain.potentials[:n]
        fired = self.brain.fired[:n]
        is_exc = self.brain.is_excitatory[:n]
        for i in range(n):
            if fired[i]:
                color = '#ffdd4a' if is_exc[i] else '#ff4a6b'
            else:
                color = potential_to_color(potentials[i], self.brain.thresholds[i])
            self.canvas_2d.itemconfigure(self.neuron_rects[i], fill=color)

    def _update_region_bars(self):
        canvas = self.region_canvas
        canvas.delete('all')
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        regions = list(Brain.REGIONS.items())
        n_regions = len(regions)
        bar_height = min(28, (ch - 20) // n_regions - 4)
        margin_left = 60
        margin_right = 80
        bar_max_w = cw - margin_left - margin_right

        for i, (name, (start, end)) in enumerate(regions):
            y = 10 + i * (bar_height + 8)
            total = end - start
            active = int(np.sum(self.brain.fired[start:end]))
            pct = active / total if total > 0 else 0
            color = REGION_COLORS[name]
            label = REGION_LABELS_CN[name]

            canvas.create_text(margin_left - 5, y + bar_height // 2,
                               text=label, fill=color, anchor='e',
                               font=('Microsoft YaHei UI', 9, 'bold'))
            canvas.create_rectangle(margin_left, y,
                                    margin_left + bar_max_w, y + bar_height,
                                    fill='#0a0a2a', outline='#1a1a3e')
            fill_w = int(pct * bar_max_w)
            if fill_w > 0:
                canvas.create_rectangle(margin_left, y,
                                        margin_left + fill_w, y + bar_height,
                                        fill=color, outline='')
            avg_pot = float(np.mean(self.brain.potentials[start:end]))
            canvas.create_text(margin_left + bar_max_w + 5, y + bar_height // 2,
                               text=f"{active}/{total} V:{avg_pot:.3f}",
                               fill=COLORS['text_dim'], anchor='w',
                               font=('Consolas', 8))

    def _update_graph(self):
        canvas = self.graph_canvas
        canvas.delete('all')
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        margin = 10
        graph_w = cw - 2 * margin
        graph_h = ch - 2 * margin

        canvas.create_rectangle(margin, margin,
                                margin + graph_w, margin + graph_h,
                                fill='#050510', outline='#1a1a3e')
        for i in range(1, 4):
            y = margin + graph_h * i / 4
            canvas.create_line(margin, y, margin + graph_w, y,
                               fill='#1a1a3e', dash=(2, 4))

        if len(self.activity_history) < 2:
            canvas.create_text(margin + graph_w // 2, margin + graph_h // 2,
                               text="等待数据...", fill=COLORS['text_dim'],
                               font=('Microsoft YaHei UI', 10))
            return

        data = self.activity_history
        n = len(data)
        max_val = max(max(data), 0.01)

        points = []
        for i, val in enumerate(data):
            x = margin + (i / max(n - 1, 1)) * graph_w
            y = margin + graph_h - (val / max_val) * graph_h
            points.append((x, y))
        if len(points) >= 2:
            flat = [coord for p in points for coord in p]
            canvas.create_line(*flat, fill=COLORS['accent'], width=1.5, smooth=True)

        for name in ['visual', 'auditory', 'motor', 'association', 'memory']:
            rdata = self.region_history[name]
            if len(rdata) < 2:
                continue
            color = REGION_COLORS[name]
            rpoints = []
            for i, val in enumerate(rdata):
                x = margin + (i / max(len(rdata) - 1, 1)) * graph_w
                y = margin + graph_h - (val / max(max_val, 0.01)) * graph_h
                rpoints.append((x, y))
            if len(rpoints) >= 2:
                flat = [coord for p in rpoints for coord in p]
                canvas.create_line(*flat, fill=color, width=1, smooth=True)

        canvas.create_text(margin + 5, margin + 5,
                           text=f"峰值: {max_val:.3f}", fill=COLORS['text_dim'],
                           anchor='nw', font=('Consolas', 8))

    def _update_status(self):
        status = self.brain.get_status()
        active = status['active']
        total_fires = status['total_fires']
        tick = status['tick']
        avg_pot = status['avg_potential']
        learn_str = "开" if status['learning'] else "关"

        self.status_label.configure(
            text=f"Tick: {tick} | 活跃: {active}/{self.brain.num_neurons} | "
                 f"总放电: {total_fires} | 学习: {learn_str}")
        self.header_status.configure(
            text=f"Tick: {tick} | 活跃: {active} | 平均电位: {avg_pot:.4f} | "
                 f"放电率: {status['avg_fire_rate']:.6f}")

    def _update_neuron_info(self, nid):
        if nid < 0 or nid >= self.brain.num_neurons:
            return
        info = self.brain.get_neuron_info(nid)
        region_cn = REGION_LABELS_CN.get(info['region'], info['region'])
        ntype_cn = '兴奋性' if info['type'] == 'excitatory' else '抑制性'
        fired_str = "⚡ 放电!" if info['fired'] else "静息"
        ref_str = f"不应期({info['refractory']})" if info['refractory'] > 0 else "就绪"

        text = (
            f"神经元 #{info['id']}\n"
            f"类型: {ntype_cn}  脑区: {region_cn}\n"
            f"膜电位: {info['potential']:.6f}  阈值: {info['threshold']:.2f}\n"
            f"状态: {fired_str}  {ref_str}\n"
            f"放电次数: {info['fire_count']}  上次: Tick {info['last_fire']}\n"
            f"输出: {info['out_strength']:.4f}  输入: {info['in_strength']:.4f}\n"
            f"存储: 2KB"
        )

        self.info_text.configure(state='normal')
        self.info_text.delete('1.0', 'end')
        self.info_text.insert('1.0', text)
        self.info_text.configure(state='disabled')

    def run(self):
        self.root.mainloop()


def main():
    app = BrainSimulatorGUI()
    app.run()


if __name__ == '__main__':
    main()
