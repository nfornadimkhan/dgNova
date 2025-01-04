from manim import *
import numpy as np
from scipy import stats
from dgNova.field_designs import UNREP

class HeatplotTransition(Scene):
    def construct(self):
        # Create instance for demonstration
        unrep_sim = UNREP(
            row=15, column=20,       
            heterogeneity=0.8,
            mean=5.3, sd=0, ne=1,
            design='moving_grid'
        )
        
        # Run analysis
        results = unrep_sim.analyze()
        
        # Title
        title = Text("Spatial Adjustment in Unreplicated Design", font_size=35)
        self.play(Write(title))
        self.wait()
        self.play(title.animate.scale(0.7).to_edge(UP))

        # Create heatplot visualization
        cell_size = 0.2
        grid_width = unrep_sim.columns * cell_size
        grid_height = unrep_sim.rows * cell_size
        start_x = -grid_width/2 + cell_size/2
        start_y = grid_height/2 - cell_size/2

        # Raw data heatplot
        raw_title = Text("Raw Field Data", font_size=25)
        raw_title.next_to(title, DOWN)
        
        raw_cells = VGroup()
        raw_matrix = unrep_sim.data
        raw_min, raw_max = np.min(raw_matrix), np.max(raw_matrix)
        
        for i in range(unrep_sim.rows):
            for j in range(unrep_sim.columns):
                value = raw_matrix[i, j]
                norm_value = (value - raw_min) / (raw_max - raw_min)
                cell = Square(side_length=cell_size)
                cell.set_fill(color=interpolate_color(BLUE, RED, norm_value), opacity=0.8)
                cell.set_stroke(WHITE, 0.5)
                cell.move_to([start_x + j*cell_size, start_y - i*cell_size, 0])
                raw_cells.add(cell)
        
        # Add color scale for raw data
        raw_scale = Rectangle(height=0.3, width=3)
        raw_scale.next_to(raw_cells, DOWN, buff=0.5)
        raw_scale.set_shading(lambda x: interpolate_color(BLUE, RED, x))
        raw_min_text = Text(f"{raw_min:.2f}", font_size=20).next_to(raw_scale, LEFT)
        raw_max_text = Text(f"{raw_max:.2f}", font_size=20).next_to(raw_scale, RIGHT)

        self.play(Write(raw_title))
        self.play(Create(raw_cells))
        self.play(
            Create(raw_scale),
            Write(raw_min_text),
            Write(raw_max_text)
        )
        self.wait(2)

        # Adjusted data heatplot
        adj_title = Text("Spatially Adjusted Data", font_size=25)
        adj_cells = VGroup()
        adj_matrix = results['adjusted_values']
        adj_min, adj_max = np.min(adj_matrix), np.max(adj_matrix)
        
        for i in range(unrep_sim.rows):
            for j in range(unrep_sim.columns):
                value = adj_matrix[i, j]
                norm_value = (value - adj_min) / (adj_max - adj_min)
                cell = Square(side_length=cell_size)
                cell.set_fill(color=interpolate_color(BLUE, RED, norm_value), opacity=0.8)
                cell.set_stroke(WHITE, 0.5)
                cell.move_to([start_x + j*cell_size, start_y - i*cell_size, 0])
                adj_cells.add(cell)

        # Add color scale for adjusted data
        adj_scale = Rectangle(height=0.3, width=3)
        adj_scale.next_to(raw_cells, DOWN, buff=0.5)
        adj_scale.set_shading(lambda x: interpolate_color(BLUE, RED, x))
        adj_min_text = Text(f"{adj_min:.2f}", font_size=20).next_to(adj_scale, LEFT)
        adj_max_text = Text(f"{adj_max:.2f}", font_size=20).next_to(adj_scale, RIGHT)

        # Show transition
        self.play(
            Transform(raw_title, adj_title),
            Transform(raw_cells, adj_cells),
            Transform(raw_scale, adj_scale),
            Transform(raw_min_text, adj_min_text),
            Transform(raw_max_text, adj_max_text)
        )
        self.wait(2)

        # Add explanation
        explanation = Text(
            "Spatial adjustment removes field trends\nand neighbor effects",
            font_size=25, line_spacing=1.5
        ).next_to(adj_scale, DOWN, buff=0.5)
        self.play(Write(explanation))
        self.wait(2)

        # Cleanup
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )

if __name__ == "__main__":
    # Command to render:
    # manim -pql basic_stats.py HeatplotTransition
    pass 