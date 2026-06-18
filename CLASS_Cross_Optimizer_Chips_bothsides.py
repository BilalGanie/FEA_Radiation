#!/usr/bin/env python
# -*- coding: utf-8 -*-
# sphinx_gallery_thumbnail_number = 3
r"""
-----------------------------------------------------------------
Modelling with convection of the MightyPixel baby13 cross section
-----------------------------------------------------------------
"""

from fileinput import filename
import time
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
 
from pygimli.solver import solve
from pygimli.viewer import show
from pygimli.viewer.mpl import drawStreams
 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches
 
import pygimli.viewer.mpl as pgmpl
pgmpl.noShow(True)
 
pg.hold(True)  # Prevents display
 
# Turn on interactive mode
plt.ion()
 
#mesh maximum size
max_mesh_size = 5.0
 
DEBUG = False
outputfolder = "results/FEA_vs_MB_Si_BothSides/"
 
plt.rcParams["figure.autolayout"] = True
 
#check if folder exists and create if not
import os
def create_output_folder_if_not_exists(outputfolder):
        if not os.path.exists(outputfolder):
                os.makedirs(outputfolder)
 
create_output_folder_if_not_exists(outputfolder)
 
# Main class for FEA study of the cross section
class FEA_study:
    def __init__(self,
                 
        modelname, tlocy=2.40, model_length=116.648,
        tiOD=2.5, tiWT=0.15,
        fthickness=4, cfthickness=0.15, ti_glue_thickness=0.1, si_glue_thickness=0.1, si_thickness=0.15,
        si_length=22.56, si_edge_length=4.95,

        # Top-down chip layout (for schematics)
        chip_columns=0, chip_gap=0.1, chip_width=20.660, chip_offset=0.0,
        edge_clearance=4.0, si_gap=16.016, si_inner_overlap=0.95,
        silicon_edge_offset=4.95,
        powerdensity=0.2e-2,
        # Thermal and radiation properties (scaled to mm units)
        fx=10 * (213.5 + 142.3) / 2, xcf=10 * 23.7, xtig=10 * 22.19, xti=10 * 3.56, xco2=33.64 * 10, xsilicon = 10*93.66, xsi_glue=10*19.4,
        tigk=2.4e-3, fk=20.3e-3, cfk=180e-3, sik=148e-3, si_gluek=0.2e-3, tik = 14e-3,

        hair=25e-6, # Heat transfer coefficient (W/(mm²·K)) - converted from 25 W/(m²·K) to mm units
        tinf=20.0, # Ambient temperature (°C)
        num_tubes=3, # Number of cooling tubes (automatically placed symmetrically)


        saveplots=True,
        zoom_region=None,
        mesh_zoom_width=2.0,
        mesh_zoom_height=0.5,):


        """
        Initialize the FEA_study object with geometry, material, and simulation parameters.
        """
        self.modelname = modelname
        self.tinf = tinf  # Ambient temperature (°C)
        self.hair = hair  # Heat transfer coefficient (W/(mm²·K))
        self.model_length = model_length  # Length of the model (mm)
        self.num_tubes = num_tubes  # Number of cooling tubes
        self.zoom_region = zoom_region
        self.mesh_zoom_width = mesh_zoom_width
        self.mesh_zoom_height = mesh_zoom_height
        #self.tlocx = model_length / 2  # Default tube x-location (center, not used for multi-tube)
        self.tlocy = tlocy  # Tube y-location (mm)
 

        self.tiOD = tiOD  # Titanium tube outer diameter (mm)
        self.tiWT = tiWT  # Titanium tube wall thickness (mm)
        self.fthickness = fthickness  # Foam thickness (mm)
        self.cfthickness = cfthickness  # Carbon fiber thickness (mm)
        self.ti_glue_thickness = ti_glue_thickness  # Glue thickness around tube (mm)
        self.gthickness = self.ti_glue_thickness  # Alias for compatibility
        self.si_glue_thickness = si_glue_thickness  # Silicon glue thickness (mm)
        self.si_thickness = si_thickness  # Silicon sensor thickness (mm)
        self.silicon_length = si_length  # Silicon sensor length (mm)
        self.edge_clearance = edge_clearance  # Left/right stave edge clearance (mm)
        self.si_gap = si_gap  # Repeated gap between silicon segments (mm)
        self.si_inner_overlap = si_inner_overlap  # Overlap between top and bottom silicon rows (mm)
        self.silicon_edge_offset = silicon_edge_offset  # Edge silicon start offset (mm)
        self.silicon_spacing = self.silicon_length + self.si_gap


        # Total model height calculation (mm)
        self.model_height = (
        2 * self.cfthickness
        + self.fthickness
        + 2 * (self.si_glue_thickness + self.si_thickness)) # Note: CF layers are on top and bottom, foam is in the middle, silicon + glue are above the top CF layer

 
        # Radiation lengths (dimensionless, same units as thicknesses)
        self.xf = fx
        self.xcf = xcf
        self.xtig = xtig
        self.xti = xti
        self.xco2 = xco2
        self.xsilicon = xsilicon  # Silicon radiation length (mm)
        self.xsi_glue = xsi_glue  # Silicon glue radiation length (mm)

        # Thermal conductivities (W/(mm·K))
        self.fk = fk
        self.cfk = cfk
        self.tik = tik  # Titanium thermal conductivity (W/(mm·K))
        self.sik = sik
        self.tigk= tigk
        self.si_gluek = si_gluek
 
        # CF-Silicon glue
        self.powerdensity = powerdensity  # Power density (W/mm^2)
        self.saveplots = saveplots  # Whether to save plots
 
        # Initialize geometry lists
        self.gfoams = []
        self.tis = []
        self.innertis = []
        # --------------------------------------------------
        # --------------------------------------------------
        # Automatic tube placement
        # --------------------------------------------------
        N = self.num_tubes
        L = self.model_length

        if N == 1:
            self.tube_positions = [L / 2.0]

        elif N == 2:
            self.tube_positions = [
                L / 4.0,
                3.0 * L / 4.0
            ]

        elif N == 3:
            self.tube_positions = [
                L / 6.0,
                L / 2.0,
                5.0 * L / 6.0
            ]

        else:
            raise ValueError(f"Unsupported number of tubes: {N}")

        min_clearance = self.tiOD / 2 + self.ti_glue_thickness

        assert self.tube_positions[0] > min_clearance, \
            "Leftmost tube too close to left edge"

        assert self.tube_positions[-1] < L - min_clearance, \
            "Rightmost tube too close to right edge"

        # Positions (as specified)
        self.chip_columns = chip_columns
        self.chip_gap = chip_gap
        self.chip_width = chip_width
        self.chip_offset = chip_offset

        if self.chip_columns and self.chip_columns > 0:
                # top silicon starting positions
                spacing = self.chip_width + self.chip_gap
                self.top_silicon_x = [self.chip_offset + i * spacing for i in range(self.chip_columns)]
                # bottom sensors offset by half spacing to create staggering (if fits)
                half_off = spacing / 2.0
                self.bot_silicon_x = [self.chip_offset + half_off + i * spacing for i in range(self.chip_columns)]
                # trim any sensors that exceed model length
                self.top_silicon_x = [x for x in self.top_silicon_x if x + self.chip_width <= self.model_length]
                self.bot_silicon_x = [x for x in self.bot_silicon_x if x + self.chip_width <= self.model_length]
                # update silicon length used for plotting geometry
                self.silicon_length = self.chip_width
        else:
                self.top_silicon_x = [0.0, 38.566, 77.132]
                self.bot_silicon_x = [16.956, 55.522, 94.088]

        # Small edge silicon/glue strips visible in the schematic
        self.bot_silicon_edge_x = []
        self.top_silicon_edge_x = []



    def calc_avg_X0(self):
        """
        Calculate average and peak radiation length X0 across the model.
        X0 is computed for tracks normal to the stave (along y),
        as a function of x.
        """

        # --------------------------------------------------
        # Helper: chord length through a circle
        # --------------------------------------------------
        def chord_length(r, dx): # r is radius, dx is horizontal distance from center
            return 2.0 * np.sqrt(np.maximum(0.0, r**2 - dx**2)) # Ensure non-negative argument for sqrt

        # --------------------------------------------------
        # Foam with cylindrical holes (tube + tube glue)
        # --------------------------------------------------
        def X0_foam(x):
            foamhole_r = self.tiOD / 2 + self.ti_glue_thickness # Radius of the cylindrical hole in the foam for the tube + glue
            x0 = np.ones_like(x) * (self.fthickness / self.xf)

            for tlocx in self.tube_positions:
                dx = x - tlocx
                mask = np.abs(dx) < foamhole_r
                x0[mask] -= chord_length(foamhole_r, dx[mask]) / self.xf

            return x0

        # --------------------------------------------------
        # Carbon fiber skins (top + bottom)
        # --------------------------------------------------
        def X0_cf(x):
            return np.ones_like(x) * (2.0 * self.cfthickness / self.xcf)

        # --------------------------------------------------
        # Glue around titanium tubes (cylindrical shell)
        # --------------------------------------------------
        def X0_ti_glue(x):
            ti_r = self.tiOD / 2
            glue_r = ti_r + self.ti_glue_thickness
            x0 = np.zeros_like(x)

            for tlocx in self.tube_positions:
                dx = x - tlocx
                mask = np.abs(dx) < glue_r
                outer = chord_length(glue_r, dx[mask])
                inner = chord_length(ti_r, dx[mask])
                x0[mask] += (outer - inner) / self.xtig

            return x0

        # --------------------------------------------------
        # Titanium tube wall
        # --------------------------------------------------
        def X0_ti(x):
            ti_r = self.tiOD / 2
            ti_inner = ti_r - self.tiWT
            x0 = np.zeros_like(x)

            for tlocx in self.tube_positions:
                dx = x - tlocx
                mask = np.abs(dx) < ti_r
                outer = chord_length(ti_r, dx[mask])
                inner = chord_length(ti_inner, dx[mask])
                x0[mask] += (outer - inner) / self.xti

            return x0

        # --------------------------------------------------
        # CO2 inside tubes
        # --------------------------------------------------
        def X0_CO2(x):
            ti_inner = (self.tiOD - 2.0 * self.tiWT) / 2.0
            x0 = np.zeros_like(x)

            for tlocx in self.tube_positions:
                dx = x - tlocx
                mask = np.abs(dx) < ti_inner
                x0[mask] += chord_length(ti_inner, dx[mask]) / self.xco2

            return x0

        # --------------------------------------------------
        # Local silicon glue (top + bottom)
        # --------------------------------------------------
        
        def X0_local_silicon_glue(x):
                x0 = np.zeros_like(x)
                for x_start in self.top_silicon_x + self.bot_silicon_x:
                        mask = (x >= x_start) & (x <= x_start + self.silicon_length)
                        x0[mask] += self.si_glue_thickness / self.xsi_glue
                for x_start in self.top_silicon_edge_x + self.bot_silicon_edge_x:
                        mask = (x >= x_start) & (x <= x_start + self.si_edge_length)
                        x0[mask] += self.si_glue_thickness / self.xsi_glue
                return x0


        # --------------------------------------------------
        # Local silicon sensors (top + bottom)
        # --------------------------------------------------
        def X0_local_silicon(x):
                x0 = np.zeros_like(x)
                for x_start in self.top_silicon_x + self.bot_silicon_x:
                        mask = (x >= x_start) & (x <= x_start + self.silicon_length)
                        x0[mask] += self.si_thickness / self.xsilicon
                for x_start in self.top_silicon_edge_x + self.bot_silicon_edge_x:
                        mask = (x >= x_start) & (x <= x_start + self.si_edge_length)
                        x0[mask] += self.si_thickness / self.xsilicon
                return x0


        # --------------------------------------------------
        # Total X0 profile
        # --------------------------------------------------
        x = np.linspace(0.0, self.model_length, 2001)

        X0 = (
            X0_foam(x)+ X0_cf(x)+ X0_ti_glue(x)
            + X0_ti(x)+ X0_CO2(x)+ X0_local_silicon_glue(x)
            + X0_local_silicon(x)
        )

        avg_X0 = np.sum(X0) / len(X0)

        if self.saveplots: 
                plt.clf()
                plt.plot(x, X0*100, label='X0(x)')
                plt.xlabel('x [mm]')
                plt.ylabel('X0 [mm]')
                plt.title(f'Average X0: {avg_X0*100:.2f} mm')
                plt.legend()
                plt.savefig(f"{outputfolder}/{self.modelname}_X0.png", dpi=300)
                #plt.xlim([self.tlocx - 3, self.tlocx + 3])
                plt.savefig(f"{outputfolder}/{self.modelname}_zoom_X0.png", dpi=300)
                plt.clf()

        self.avg_X0 = avg_X0
        self.peak_X0 = np.max(X0)


    def create_geometry(self):
        """Create geometry for the MightyPixel baby13 cross section"""

        foamradius = self.tiOD/2 + self.gthickness
        tiradius = self.tiOD/2
        innertiradius = tiradius - self.tiWT

        pheight = 0

        # Bottom silicon sensor blocks (localized)
        self.bottom_silicon_sensors = []
        self.bottom_silicon_glues = []
        

        for x0 in self.bot_silicon_x:
                sensor = mt.createRectangle(
                        start=[x0, 0],
                        end=[x0 + self.silicon_length, self.si_thickness],
                        boundaryMarker=[101,102,103,104],
                        marker=9,
                        area=0.01)
                glue = mt.createRectangle(
                        start=[x0, self.si_thickness],
                        end=[x0 + self.silicon_length, self.si_thickness + self.si_glue_thickness],
                        boundaryMarker=[5,6,7,8], marker=10, area=0.01)
                self.bottom_silicon_sensors.append(sensor)
                self.bottom_silicon_glues.append(glue)

        for x0 in self.bot_silicon_edge_x:
                sensor = mt.createRectangle(
                start=[x0, 0],
                end=[x0 + self.si_edge_length, self.si_thickness],
                boundaryMarker=[101,102,103,104],
                marker=9,
                area=0.01)
                glue = mt.createRectangle(
                        start=[x0, self.si_thickness],
                        end=[x0 + self.si_edge_length, self.si_thickness + self.si_glue_thickness],
                        boundaryMarker=[5,6,7,8], marker=10, area=0.01)
                self.bottom_silicon_edge_sensors.append(sensor)
                self.bottom_silicon_edge_glues.append(glue)

        pheight = self.si_thickness + self.si_glue_thickness

        # Bottom carbon fiber layer
        self.bottom_carbon_fiber = mt.createRectangle(
                start=[0, pheight],
                end=[self.model_length, pheight + self.cfthickness],
                boundaryMarker=[9,10,11,12], marker=1, area=0.01)
        pheight += self.cfthickness

        # Foam layer with tube region cut-out
        self.foam = mt.createRectangle(
                start=[0, pheight],
                end=[self.model_length, pheight + self.fthickness],
                boundaryMarker=[13,14,15,16], marker=2, area=0.01)

        foam_bottom = pheight
        half_foam = self.fthickness / 2.0
        self.foam_end_cut_holes = []

        # Left end cut: remove LOWER half of the foam over the 4 mm edge clearance
        self.foam_end_cut_holes.append(
            mt.createPolygon(
                [[0.0, foam_bottom],
                [self.edge_clearance, foam_bottom],
                [self.edge_clearance, foam_bottom + half_foam],
                [0.0, foam_bottom + half_foam]],
                isHole=True,
                isClosed=True)
        )

        # Right end cut: remove UPPER half of the foam over the 4 mm edge clearance
        self.foam_end_cut_holes.append(
            mt.createPolygon(
                [[self.model_length - self.edge_clearance, foam_bottom + half_foam],
                [self.model_length, foam_bottom + half_foam],
                [self.model_length, foam_bottom + self.fthickness],
                [self.model_length - self.edge_clearance, foam_bottom + self.fthickness]],
                isHole=True,
                isClosed=True)
        )

        pheight += self.fthickness


        # Top carbon fiber layer
        self.top_carbon_fiber = mt.createRectangle(
                start=[0, pheight],
                end=[self.model_length, pheight + self.cfthickness],
                boundaryMarker=[17,18,19,20], marker=1, area=0.01)
        pheight += self.cfthickness



        self.top_silicon_glues = []
        self.top_silicon_sensors = []

        for x0 in self.top_silicon_x:
            glue = mt.createRectangle(
                start=[x0, pheight],
                end=[x0 + self.silicon_length, pheight + self.si_glue_thickness],
                boundaryMarker=[21, 22, 23, 24],
                marker=7,
                area=0.01
            )

            sensor = mt.createRectangle(
                start=[x0, pheight + self.si_glue_thickness],
                end=[x0 + self.silicon_length,
                    pheight + self.si_glue_thickness + self.si_thickness],
                boundaryMarker=[201, 202, 203, 204],
                marker=8,
                area=0.01
            )

            self.top_silicon_glues.append(glue)
            self.top_silicon_sensors.append(sensor)

        pheight += self.si_glue_thickness + self.si_thickness


        # --- Tube geometry ---
        self.tube_glue_shells = []
        self.titanium_tubes = []
        self.co2_coolant_holes = []

        for tlocx in self.tube_positions:
                gfoamxys = []
                tixys = []
                innertixys = []

                for i in range(100):
                        angle = i * np.pi / 50
                        gfoamxys.append([
                                tlocx + foamradius * np.cos(angle),
                                self.tlocy + foamradius * np.sin(angle)
                        ])
                        tixys.append([
                                tlocx + tiradius * np.cos(angle),
                                self.tlocy + tiradius * np.sin(angle)
                        ])
                        innertixys.append([
                                tlocx + innertiradius * np.cos(angle),
                                self.tlocy + innertiradius * np.sin(angle)
                        ])

                self.tube_glue_shells.append(
                        mt.createPolygon(gfoamxys, marker=3, isClosed=True, area=0.01)
                )
                self.titanium_tubes.append(
                        mt.createPolygon(tixys, marker=4, isClosed=True, area=0.01)
                )
                self.co2_coolant_holes.append(
                        mt.createPolygon(innertixys, isHole=True, isClosed=True, boundaryMarker=99)
                )

        # Final geometry assembly
        self.geom = (
            self.bottom_carbon_fiber
            + self.foam
            + self.top_carbon_fiber
        )

        for part in (
            self.tube_glue_shells
            + self.titanium_tubes
            + self.co2_coolant_holes
            + self.foam_end_cut_holes
            + self.bottom_silicon_sensors
            + self.bottom_silicon_glues
            + self.top_silicon_glues
            + self.top_silicon_sensors
        ):
            self.geom += part

   
    def plot_geometry(self):
        """ Plot the geometry using pygimli's show function. Markers are shown for better visualization of boundaries. """
        if not self.saveplots:
                return

        # Save the standard (inscale) view for reference
        ax, cb = pg.show(self.geom, markers=True, aspect='auto')
        plt.savefig(f"{outputfolder}/{self.modelname}_inscale_geometry.png", dpi=300)

        # Save a zoomed-in view of the region of interest if requested
        if self.zoom_region is not None:
                xmin, xmax, ymin, ymax = self.zoom_region
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                plt.savefig(f"{outputfolder}/{self.modelname}_geometry_zoom.png", dpi=300)
        plt.clf()

        # Clear and save an equal-aspect view with annotated layer boundaries
        ax, cb = pg.show(self.geom, markers=True, aspect='equal')

        # Force axis limits to the model extents so vertical scale is correct
        try:
                ax.set_xlim(0, self.model_length)
                ax.set_ylim(0, self.model_height)
        except Exception:
                pass

        # Annotate layer boundary positions (y-values)
        bottom_sensor_top = self.si_thickness
        bottom_glue_top = self.si_thickness + self.si_glue_thickness
        bottom_cf_top = self.si_thickness + self.si_glue_thickness + self.cfthickness
        foam_top = bottom_cf_top + self.fthickness
        top_cf_top = foam_top + self.cfthickness
        top_glue_top = top_cf_top + self.si_glue_thickness
        top_sensor_top = top_glue_top + self.si_thickness

        lines = [
                (bottom_sensor_top, 'bottom sensor'),
                (bottom_glue_top, 'bottom glue'),
                (bottom_cf_top, 'bottom CF'),
                (foam_top, 'foam top'),
                (top_cf_top, 'top CF'),
                (top_glue_top, 'top glue'),
                (top_sensor_top, 'top sensor'),
        ]

        for y, label in lines:
                ax.hlines(y, 0, self.model_length, colors='k', linestyles='--', linewidth=0.5)
                ax.text(0.5, y + 0.01 * self.model_height, label, verticalalignment='bottom', fontsize=8, color='k')

        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        plt.savefig(f"{outputfolder}/{self.modelname}_geometry_equal.png", dpi=300, bbox_inches='tight')


    def plot_schematic(self):
        """Save a schematic drawing with dimensions, overlaps and tube sizes."""
        if not self.saveplots:
                return

        fig, ax = plt.subplots(figsize=(10, 3))

        # vertical positions matching model stacking
        bottom_y = 0.0
        bot_sensor_h = self.si_thickness
        bot_glue_h = self.si_glue_thickness
        bot_cf_h = self.cfthickness
        foam_h = self.fthickness
        top_cf_h = self.cfthickness
        top_glue_h = self.si_glue_thickness
        top_sensor_h = self.si_thickness

        # draw bottom sensors and glues
        for x0 in self.bot_silicon_x:
                rect = patches.Rectangle((x0, bottom_y), self.silicon_length, bot_sensor_h,
                                         facecolor='lightblue', edgecolor='k', alpha=0.8)
                ax.add_patch(rect)
                glue = patches.Rectangle((x0, bottom_y + bot_sensor_h), self.silicon_length, bot_glue_h,
                                          facecolor='lightsteelblue', edgecolor='k', alpha=0.6)
                ax.add_patch(glue)
        for x0 in self.bot_silicon_edge_x:
                rect = patches.Rectangle((x0, bottom_y), self.si_edge_length, bot_sensor_h,
                                         facecolor='lightblue', edgecolor='k', alpha=0.8)
                ax.add_patch(rect)
                glue = patches.Rectangle((x0, bottom_y + bot_sensor_h), self.si_edge_length, bot_glue_h,
                                          facecolor='lightsteelblue', edgecolor='k', alpha=0.6)
                ax.add_patch(glue)

        # bottom CF
        y_bot_cf = bottom_y + bot_sensor_h + bot_glue_h
        cf_rect = patches.Rectangle((0, y_bot_cf), self.model_length, bot_cf_h, facecolor='lightgray', edgecolor='k', alpha=0.5)
        ax.add_patch(cf_rect)

        # foam
        y_foam = y_bot_cf + bot_cf_h
        foam_rect = patches.Rectangle((0, y_foam), self.model_length, foam_h, facecolor='wheat', edgecolor='k', alpha=0.4)
        ax.add_patch(foam_rect)

        # top CF
        y_top_cf = y_foam + foam_h
        cf_rect2 = patches.Rectangle((0, y_top_cf), self.model_length, top_cf_h, facecolor='lightgray', edgecolor='k', alpha=0.5)
        ax.add_patch(cf_rect2)

        # top sensors and glues
        y_top_glue = y_top_cf + top_cf_h
        for x0 in self.top_silicon_x:
                glue = patches.Rectangle((x0, y_top_glue), self.silicon_length, top_glue_h, facecolor='lightsteelblue', edgecolor='k', alpha=0.6)
                ax.add_patch(glue)
                rect = patches.Rectangle((x0, y_top_glue + top_glue_h), self.silicon_length, top_sensor_h, facecolor='lightgreen', edgecolor='k', alpha=0.8)
                ax.add_patch(rect)
        for x0 in self.top_silicon_edge_x:
                glue = patches.Rectangle((x0, y_top_glue), self.si_edge_length, top_glue_h, facecolor='lightsteelblue', edgecolor='k', alpha=0.6)
                ax.add_patch(glue)
                rect = patches.Rectangle((x0, y_top_glue + top_glue_h), self.si_edge_length, top_sensor_h, facecolor='lightgreen', edgecolor='k', alpha=0.8)
                ax.add_patch(rect)

        # draw tubes (as circles) and annotate diameters
        tiradius = self.tiOD / 2.0
        for tlocx in self.tube_positions:
                circ = patches.Circle((tlocx, self.tlocy), tiradius, edgecolor='tab:orange', facecolor='none', linewidth=1.2)
                ax.add_patch(circ)
                ax.text(tlocx, self.tlocy - tiradius - 0.1, f"Ti OD={self.tiOD:.2f} mm", ha='center', va='top', fontsize=7, color='tab:orange')

        # compute and annotate horizontal overlaps between top and bottom sensors,
        # including both full 22.56 mm sensors and 4.95 mm edge sensors

        top_sensor_regions = (
            [(x0, self.silicon_length) for x0 in self.top_silicon_x]
        
        )

        bottom_sensor_regions = (
            [(x0, self.silicon_length) for x0 in self.bot_silicon_x] 
                )

        for tx, t_len in top_sensor_regions:
            t_left = tx
            t_right = tx + t_len

            for bx, b_len in bottom_sensor_regions:
                b_left = bx
                b_right = bx + b_len

                left = max(t_left, b_left)
                right = min(t_right, b_right)

                if right > left:
                        mid = (left + right) / 2.0
                        overlap = right - left

                        ax.plot(
                                [left, right],
                                [y_top_glue + top_glue_h + top_sensor_h + 0.05] * 2,
                                color='red',
                                linewidth=2
                        )

                        ax.text(
                                mid,
                                y_top_glue + top_glue_h + top_sensor_h + 0.12,
                                f"{overlap:.3f} mm",
                                ha='center',
                                fontsize=7,
                                color='red'
                        )

                # annotate vertical thicknesses
                ax.text(self.model_length*0.02, bottom_y + bot_sensor_h/2, f"bot sensor {bot_sensor_h:.2f} mm", va='center', fontsize=7)
                ax.text(self.model_length*0.02, bottom_y + bot_sensor_h + bot_glue_h/2, f"bot glue {bot_glue_h:.2f} mm", va='center', fontsize=7)
                ax.text(self.model_length*0.02, y_bot_cf + bot_cf_h/2, f"CF {bot_cf_h:.2f} mm", va='center', fontsize=7)
                ax.text(self.model_length*0.02, y_foam + foam_h/2, f"foam {foam_h:.2f} mm", va='center', fontsize=7)
                ax.text(self.model_length*0.02, y_top_glue + top_glue_h/2, f"top glue {top_glue_h:.2f} mm", va='center', fontsize=7)
                ax.text(self.model_length*0.02, y_top_glue + top_glue_h + top_sensor_h/2, f"top sensor {top_sensor_h:.2f} mm", va='center', fontsize=7)

        ax.set_xlim(-5, min(self.model_length + 5, 130))
        ax.set_ylim(-0.5, self.model_height + 1.0)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f"Schematic: {self.modelname}")
        plt.tight_layout()
        plt.savefig(f"{outputfolder}/{self.modelname}_schematic.png", dpi=300, bbox_inches='tight')
        plt.clf()

    def createmesh(self):
        """ Create a mesh from the geometry using pygimli's meshtools. The quality parameter can be adjusted for finer or coarser meshes. """

        self.mesh = mt.createMesh(self.geom, quality=20) # 33 is standard for good quality
        
    def study_solve(self):
        """ Solve the PDE using pygimli's solve function. The boundary conditions are specified as dictionaries for Dirichlet, Neumann, and Robin conditions.
        The conductivity values for different regions are also provided (a). The solution is then visualized using the show function, and the maximum temperature is printed. """

        # Boundary conditions: Dirichlet on tube coolant boundary (99) if used,
        # Neumann: apply supplied heat flux to all foam outer boundaries (13-16) instead of only one marker
        # Robin: convection on all exterior CF/foam edges and on the titanium tube wall (marker 4).
        dirichletBC = {99: 0}  # Fixed coolant temperature on inner tube boundary (if desired)
        # Apply heating flux symmetrically on the foam outer edges (markers 13-16)
        # neumanBC = {
        #         104: self.powerdensity,  # top surface of bottom silicon, towards glue
        #         203: self.powerdensity   # bottom surface of top silicon, towards glue
        # }
        # Apply convective Robin BC to outer structure edges and tube wall
        robinBC = {
            9: (self.hair, self.tinf), 10: (self.hair, self.tinf), 11: (self.hair, self.tinf), 12: (self.hair, self.tinf),
            13: (self.hair, self.tinf), 14: (self.hair, self.tinf), 15: (self.hair, self.tinf), 16: (self.hair, self.tinf),
            17: (self.hair, self.tinf), 18: (self.hair, self.tinf), 19: (self.hair, self.tinf), 20: (self.hair, self.tinf),
            4: (self.hair, self.tinf)
        }

        ###############################################################################
        # The boundary conditions are passed using the bc keyword dictionary.
        source = pg.Vector(self.mesh.cellCount(), 0.0)

        for cell in self.mesh.cells():
                if cell.marker() in [8, 9]:   # top and bottom silicon
                        source[cell.id()] = self.powerdensity / self.si_thickness

        u = solve(
    self.mesh,
    bc={'Dirichlet': dirichletBC, 'Robin': robinBC},
    a={
        0: self.fk,
        1: self.cfk,
        2: self.fk,
        3: self.tigk,
        4: self.tik,
        7: self.si_gluek,
        8: self.sik,
        9: self.sik,
        10: self.si_gluek
    },
    f=source
)
        # Maximum temperature from u
        self.max_temperature = np.max(u)
        
        print(f"Maximum temperature: {self.max_temperature:.2f} K")

        # Thermal symmetry / geometry check
        H = self.model_height
        L = self.model_length

        print("Model length:", L)
        print("Model height:", H)
        print("Tube positions:", self.tube_positions)
        print("Top silicon:", self.top_silicon_x, self.top_silicon_edge_x)
        print("Bottom silicon:", self.bot_silicon_x, self.bot_silicon_edge_x)
        print(f"Maximum temperature: {self.max_temperature:.2f} K") 

        if not self.saveplots:
                return

        cmin = np.min(u)
        cmax = np.max(u)

        # ax, cbar = show(self.mesh, data=u, xlabel="x[mm]", ylabel="y[mm]", nLevs=10, cmap="Spectral_r", showMesh=True, cMin=cmin, cMax=cmax)#, aspect='auto')# levels=np.linspace(min(u), max(u), 14), cMap='viridis', colorBar=True, label='Solution $u$')
        fig, ax = plt.subplots(figsize=(18,5))

        ax, cbar = show(
        self.mesh,
        data=u,
        ax=ax,
        xlabel="x [mm]",
        ylabel="y [mm]",
        nLevs=20,
        cmap="Spectral_r",
        showMesh=False,
        cMin=cmin,
        cMax=cmax
        )

        ax.set_xlim(0, self.model_length)
        ax.set_ylim(-0.5, self.model_height + 0.5)

        ax.set_aspect('auto')

        plt.tight_layout()

        plt.savefig(
        f"{outputfolder}/{self.modelname}_solution_large.png",
        dpi=600,
        bbox_inches='tight'
        )
        # Large meshed thermal plot for thesis
        fig, ax = plt.subplots(figsize=(14, 4))

        ax, cbar = show(
        self.mesh,
        data=u,
        ax=ax,
        xlabel="x [mm]",
        ylabel="y [mm]",
        nLevs=20,
        cmap="Spectral_r",
        showMesh=True,
        cMin=cmin,
        cMax=cmax
        )

        ax.set_xlim(0, self.model_length)
        ax.set_ylim(0, self.model_height)
        ax.set_aspect('auto')

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        cbar.set_label(r'$\Delta T$ [K]')

        plt.tight_layout()
        plt.savefig(
        f"{outputfolder}/{self.modelname}_solution_large_with_mesh.png",
        dpi=600,
        bbox_inches="tight"
        )
        plt.clf()
        

        ax, cbar = show(self.mesh, data=u, xlabel="x[mm]", ylabel="y[mm]", nLevs=10, cmap="Spectral_r", showMesh=False, cMin=cmin, cMax=cmax)#, aspect='auto')# levels=np.linspace(min(u), max(u), 14), cMap='viridis', colorBar=True, label='Solution $u$')
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        cbar.set_label(r'$\Delta T(K)$')
        plt.savefig(f"{outputfolder}/{self.modelname}_solution.png", dpi=300)


        plt.clf()

        ax, cbar = show(self.mesh, data=u, xlabel="x[mm]", ylabel="y[mm]", nLevs=10, cmap="Spectral_r", showMesh=True, cMin=cmin, cMax=cmax)# levels=np.linspace(min(u), max(u), 14), cMap='viridis', colorBar=True, label='Solution $u$')

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_xlim([self.tube_positions[0] - self.mesh_zoom_width/2.0, self.tube_positions[0] + self.mesh_zoom_width/2.0])
        ax.set_ylim([self.tlocy - self.mesh_zoom_height/2.0, self.tlocy + self.mesh_zoom_height/2.0])

        cbar.set_label(r'$\Delta T(K)$')

        #ax.set_ylim([0, 150])

        plt.savefig(f"{outputfolder}/{self.modelname}_withmesh_zoom_solution.png", dpi=300, bbox_inches='tight')

        plt.clf()

        ax, cbar = show(self.mesh, data=u, xlabel="x[mm]", ylabel="y[mm]", nLevs=10, cmap="Spectral_r", showMesh=True, cMin=cmin, cMax=cmax)# levels=np.linspace(min(u), max(u), 14), cMap='viridis', colorBar=True, label='Solution $u$')

        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_xlim([self.tube_positions[-1] - self.mesh_zoom_width/2.0, self.tube_positions[-1] + self.mesh_zoom_width/2.0])
        ax.set_ylim([self.tlocy - self.mesh_zoom_height/2.0, self.tlocy + self.mesh_zoom_height/2.0])

        cbar.set_label(r'$\Delta T(K)$')

        #ax.set_ylim([0, 150])

        plt.savefig(f"{outputfolder}/{self.modelname}_withmesh_zoom2_solution.png", dpi=300, bbox_inches='tight')


        
    def __del__(self):
        plt.close('all')

if __name__ == "__main__":

        scenarios = [
        ]

        fthicknesses = np.linspace(3, 5, 11)
        cfthicknesses = np.linspace(0.05, 0.2, 4)
        tiODs = np.linspace(1.0, 3.0, 16)
        tiWT = 0.15
        gthickness = 0.1


        for fthickness in fthicknesses:
                for cfthickness in cfthicknesses:
                        for tiOD in tiODs:
                                if fthickness - tiOD - 2*gthickness < 0.5: continue
                                for num_tubes in [1, 2, 3]:
                                        scenarios.append({
                                                "modelname": f"Cross_Optimizer_n{num_tubes}_f{fthickness:.2f}_cf{cfthickness:.2f}_tiOD{tiOD:.2f}",
                                                "fthickness": fthickness,
                                                "cfthickness": cfthickness,
                                                "tlocy": 0.15 + 0.1 + cfthickness + fthickness/2,
                                                "powerdensity": 0.4e-2,
                                                "tiOD": tiOD,
                                                "num_tubes": num_tubes,
                                                "saveplots": True,
                                        })

        print(f"Running parameter sweep for {len(scenarios)} scenarios...")

        #tlocy=2.15, tlocx=30, 
        #tiOD=2.5, tiWT=0.15, 
        #fthickness=4, cfthickness=0.15, gthickness=0.1, 
        #powerdensity=0.2e-2,
        #fx = 10*(213.5+142.3)/2, xcf=10*23.7, xg=10*22.19, xti=10*85.4,
        #gk=2.4e-3, fk=20.3e-3, cfk=180e-3, hair=25e-6, tinf=20.0):

        _time0 = time.time()

        x0s = []
        DTs = []
        peakX0s = []

        best_model = None
        best_estimator = 0.

        mega_dict = {}

        id = 0
        nfailed = 0
        for scenario in scenarios:
                print(f"Running scenario: {scenario['modelname']}")
                #elapsed time
                print(f"Elapsed time: {time.time() - _time0:.2f} s")
                #estimated total time remaining                _time = time.time()
                elapsed_time = time.time() - _time0
                scenarios_done = scenarios.index(scenario) + 1
                total_scenarios = len(scenarios)
                estimated_total_time = elapsed_time / scenarios_done * (total_scenarios - id)
                print(f"Estimated total time remaining: {estimated_total_time:.2f} s")
                print(f"Estimated total time remaining: {estimated_total_time/3600:.2f} h")
                try:
                        a = FEA_study(**scenario)
                        a.create_geometry()
                        a.plot_geometry()
                        a.createmesh()
                        a.study_solve()
                        a.calc_avg_X0()


                except Exception as e:
                        nfailed += 1
                        print(f"Error in scenario: {scenario['modelname']}")
                        print(e)
                        raise

                #Fraction of failed scenarios
                print(f"Failed scenarios: {nfailed}/{id+1} ({nfailed/(id+1)*100:.2f}%)")

                print(f"Average X0: {a.avg_X0*100:.2f} mm")
                print(f"Peak X0: {a.peak_X0*100:.2f} mm")
                print(f"Maximum DT: {a.max_temperature:.2f} K")

                x0s.append(a.avg_X0*100)
                DTs.append(a.max_temperature)
                peakX0s.append(a.peak_X0*100)

                #adding to megadict
                mega_dict[id] = scenario.copy()
                mega_dict[id].update({
                        "avg_X0": a.avg_X0*100,
                        "peak_X0": a.peak_X0*100,
                        "max_DT": a.max_temperature,
                })
                id += 1

                del(a)

        #save mega_dict to pickle
        import pickle
        with open(f"{outputfolder}/Cross_Optimizer_400mWcm2_results.pkl", "wb") as f:
                pickle.dump(mega_dict, f)
