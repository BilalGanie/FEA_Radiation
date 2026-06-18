#!/usr/bin/env python
# -*- coding: utf-8 -*-
# sphinx_gallery_thumbnail_number = 3
r"""
-----------------------------------------------------------------
Modelling with convection of the MightyPixel baby13 cross section
-----------------------------------------------------------------
"""

import time
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
import pprint

from pygimli.solver import solve
from pygimli.viewer import show
from pygimli.viewer.mpl import drawStreams

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import pygimli.viewer.mpl as pgmpl
pgmpl.noShow(True)

#pg.hold(True)  # Prevents display

# Turn on interactive mode
plt.ioff()

#mesh maximum size
max_mesh_size = 5.0

DEBUG = False
outputfolder = "results\\MP\\"

plt.rcParams["figure.autolayout"] = True

#check if folder exists and create if not
import os
def create_output_folder_if_not_exists(outputfolder):
        if not os.path.exists(outputfolder):
                os.makedirs(outputfolder)

create_output_folder_if_not_exists(outputfolder)

class FEA_study:
        def __init__(self, modelname, 
                     tlocy=2.15, tlocx=30, 
                     tiOD=2.5, tiWT=0.15, 
                     fthickness=4, cfthickness=0.15, gthickness=0.1, 
                     powerdensity=0.2e-2,
                     fx = 10*(213.5+142.3)/2, xcf=10*23.7, xg=10*22.19, xti=10*3.56, xco2=33.64*10,
                     gk=2.4e-3, fk=20.3e-3, cfk=180e-3, hair=25e-6, tinf=20.0,
                     saveplots=False):
                
                self.modelname = modelname
                self.tinf = tinf
                self.hair = hair # W/(m^2 K) (engineering toolbox)
                self.tlocy = tlocy
                self.tlocx = tlocx
                
                self.tiOD = tiOD
                self.tiWT = tiWT
                
                self.fthickness = fthickness
                self.cfthickness = cfthickness
                self.gthickness = gthickness

                self.gk = gk
                self.fk = fk
                self.cfk = cfk

                self.powerdensity = powerdensity

                self.xf = fx
                self.xcf = xcf
                self.xg = xg
                self.xti = xti
                self.xco2 = xco2
                self.saveplots = saveplots

       
        def print_params(self):
            pprint.pprint(self.__dict__)

#         # def calc_avg_X0(self):
#         #         def X0_foam(x):
#         #                 foamhole_r = self.tiOD/2 + self.gthickness
#         #                 x0 = np.ones_like(x)*self.fthickness/self.xf # base calculation
#         #                 #removing the foam hole
#         #                 filter = np.abs(x - self.tlocx) < foamhole_r
#         #                 x0[filter] -= (foamhole_r**2 - (x[filter] - self.tlocx)**2)**0.5 / self.xf
#         #                 return x0

#         #         def X0_cf(x):
#         #                 x0 = np.ones_like(x)*2*self.cfthickness/self.xcf # base calculation
#         #                 return x0

#         #         def X0_glue(x):
#         #                 glue_inner_r = self.tiOD/2
#         #                 glue_outer_r = glue_inner_r + self.gthickness
#         #                 x0 = np.zeros_like(x)
#         #                 x0[np.abs(x - self.tlocx) < glue_outer_r] = self.gthickness/self.xg # base calculation
#         #                 x0[np.abs(x - self.tlocx) < glue_inner_r] -= self.gthickness/self.xg # removing the part where glue is not present due to the tube
#         #                 return x0

#         #         def X0_ti(x):
#         #                 tiOD_r = self.tiOD/2
#         #                 tiID_r = tiOD_r - self.tiWT
#         #                 x0 = np.zeros_like(x)
#         #                 filter = np.abs(x - self.tlocx) < tiOD_r
#         #                 x0[filter] = (tiOD_r**2 - (x[filter] - self.tlocx)**2)**0.5 / self.xti
#         #                 filter = np.abs(x - self.tlocx) < tiID_r
#         #                 x0[filter] -= (tiID_r**2 - (x[filter] - self.tlocx)**2)**0.5 / self.xti
#         #                 return x0

#         #         def X0_CO2(x):
#         #                 x0 = np.zeros_like(x)
#         #                 tiID = self.tiOD - 2*self.tiWT
#         #                 tiID_r = tiID/2
#         #                 filter = np.abs(x - self.tlocx) < tiID_r
#         #                 x0[filter] = (tiID_r**2 - (x[filter] - self.tlocx)**2)**0.5 / self.xco2
#         #                 return x0

#         #         #x = np.linspace(0, 60, 60001)
#         #         x = np.linspace(0, 60, 200001)
#         #         X0 = X0_foam(x) + X0_cf(x) + X0_glue(x) + X0_ti(x) + X0_CO2(x)

#         #         avg_X0 = np.sum(X0) / len(X0)

#         #         if self.saveplots: 
#         #                 plt.clf()
#         #                 plt.plot(x, X0*100, label='X0(x)')
#         #                 plt.xlabel('x [mm]')
#         #                 plt.ylabel('X0 [mm]')
#         #                 plt.title(f'Average X0: {avg_X0*100:.2f} mm')
#         #                 plt.legend()
#         #                 plt.savefig(f"{outputfolder}\{self.modelname}_X0.png", dpi=300)
#         #                 plt.xlim([self.tlocx - 3, self.tlocx + 3])
#         #                 plt.savefig(f"{outputfolder}\{self.modelname}_zoom_X0.png", dpi=300)
#         #                 plt.clf()

#         #         self.avg_X0 = avg_X0
#         #         self.peak_X0 = np.max(X0)


#         # def create_geometry(self):
#         #         """Create geometry for the MightyPixel baby13 cross section"""

#         #         foamradius = self.tiOD/2 + self.gthickness
#         #         tiradius = self.tiOD/2
#         #         innertiradius = tiradius - self.tiWT

#         #         pheight = 0

#         #         self.CFbot = mt.createRectangle(start=[0, 0], 
#         #                                         end=[60, self.cfthickness], 
#         #                                         boundaryMarker=[1,2,3,4], marker=1, area=0.001)
#         #         pheight += self.cfthickness
#         #         self.cfoam = mt.createRectangle(start=[0, pheight], 
#         #                                         end=[60, pheight + self.fthickness], 
#         #                                         boundaryMarker=[5,6,7,8], marker=2, area=0.01)
#         #         pheight += self.fthickness
#         #         self.CFtop = mt.createRectangle(start=[0, pheight], 
#         #                                         end=[60, pheight + self.cfthickness], 
#         #                                         boundaryMarker=[9,10,11,12], marker=1, area=0.001)
#         #         pheight += self.cfthickness
#         #         self.glue = mt.createRectangle(start=[0, pheight],
#         #                                         end=[60, pheight + self.gthickness],
#         #                                         boundaryMarker=[17,18,19,20], marker=6, area=0.001)
#         #         pheight += self.gthickness
#         #         self.kaptop = mt.createRectangle(start=[0, pheight],
#         #                                          end=[60, pheight + self.cfthickness],
#         #                                          boundaryMarker=[13,14,15,16], marker=5, area=0.001)

#         #         gfoamxys = []
#         #         tixys = []
#         #         innertixys = []
#         #         for i in range(100):
#         #                 gfoamxys.append([self.tlocx + foamradius*np.cos(i*np.pi/50), self.tlocy + foamradius*np.sin(i*np.pi/50)])
#         #                 tixys.append([self.tlocx + tiradius*np.cos(i*np.pi/50), self.tlocy + tiradius*np.sin(i*np.pi/50)])
#         #                 innertixys.append([self.tlocx + innertiradius*np.cos(i*np.pi/50), self.tlocy + innertiradius*np.sin(i*np.pi/50)])

#         #         self.gfoam = mt.createPolygon(gfoamxys, marker=3, isClosed=True, area=0.01)
#         #         self.ti = mt.createPolygon(tixys, marker=4, isClosed=True, area=0.01)
#         #         self.innerti = mt.createPolygon(innertixys, isHole=True, isClosed=True, boundaryMarker=99)
                
#         #         self.geom = self.cfoam + self.gfoam + self.ti + self.innerti + self.CFbot + self.CFtop+ self.kaptop+ self.glue

#         # def plot_geometry(self):
#         #         """ Plot the geometry using pygimli's show function. Markers are shown for better visualization of boundaries. """
#         #         if not self.saveplots:
#         #                 return
#         #         ax, cb = pg.show(self.geom, markers=True, aspect='auto')#, figsize=(12, 13))

#         #         #ax.set_ylim([0, 150])
#         #         #ax.set_xlim([15, 25])

#         #         plt.savefig(f"{outputfolder}/{self.modelname}_geometry.png", dpi=300)

#         #         plt.clf()
#         #         ax, cb = pg.show(self.geom, markers=True)#, figsize=(12, 13))

#         #         #ax.set_ylim([0, 150])
#         #         #ax.set_xlim([15, 25])

#         #         plt.savefig(f"{outputfolder}/{self.modelname}_inscale_geometry.png", dpi=300)

#         # def createmesh(self):
#         #         """ Create a mesh from the geometry using pygimli's meshtools. The quality parameter can be adjusted for finer or coarser meshes. """

#         #         self.mesh = mt.createMesh(self.geom, quality=10) # 33 is standard for good quality
        
#         # def study_solve(self):
#         #         """ Solve the PDE using pygimli's solve function. The boundary conditions are specified as dictionaries for Dirichlet, Neumann, and Robin conditions.
#         #         The conductivity values for different regions are also provided (a). The solution is then visualized using the show function, and the maximum temperature is printed. """

#         #         # Boundary conditions: Dirichlet on tube boundaries (2) and Neumann on world boundaries (1)
#         #         dirichletBC = {99: 0} # Fixed temperature (e.g., 0 K) on tube boundaries
#         #         neumanBC = {14: self.powerdensity} # No heat flux (insulated) on world boundaries
#         #         robinBC = {14: (self.hair, self.tinf), 4:(self.hair, self.tinf)} # Robin BC on world boundaries with h=12.122e-6 (engineering toolbox) and T_inf=32 K

#         #         ###############################################################################
#         #         # The boundary conditions are passed using the bc keyword dictionary.
#         #         u = solve(self.mesh, #0.004/(31.16e-3*4.3), 
#         #                 bc={'Dirichlet': dirichletBC, "Neumann": neumanBC, "Robin": robinBC}, 
#         #                 a={1:180e-3, 2:20e-3, 3:2.4e-3, 4:14e-3, 5:0.2e-3, 6:0.2e-3})

#         #         # Maximum temperature from u
#         #         self.max_temperature = np.max(u)
#         #         print(f"Maximum temperature: {self.max_temperature:.2f} K") 

#         #         if not self.saveplots:
#         #                 return

#         #         ax, cbar = show(self.mesh, data=u, xlabel="x[mm]", ylabel="y[mm]", nLevs=10, cmap="Spectral_r", showMesh=True)#, aspect='auto')# levels=np.linspace(min(u), max(u), 14), cMap='viridis', colorBar=True, label='Solution $u$')

#         #         #plot using directly matplotlib u in the mesh points
#         #         #plt.scatter(self.mesh.cellCenters()[0], self.mesh.cellCenters()[1], c=u, cmap="Spectral_r", s=10)
#         #         #plt.colorbar(label=r'$\Delta T(K)$')

#         #         ax.set_xlabel("x [mm]")
#         #         ax.set_ylabel("x [mm]")

#         #         cbar.set_label(r'$\Delta T(K)$')

#         #         #ax.set_ylim([0, 150])

#         #         plt.savefig(f"{outputfolder}\{self.modelname}_wmesh_solution.png", dpi=300)

#         #         plt.clf()

#         #         ax, cbar = show(self.mesh, data=u, xlabel="x[mm]", ylabel="y[mm]", nLevs=10, cmap="Spectral_r", showMesh=False)#, aspect='auto')# levels=np.linspace(min(u), max(u), 14), cMap='viridis', colorBar=True, label='Solution $u$')
#         #         ax.set_xlabel("x [mm]")
#         #         ax.set_ylabel("x [mm]")
#         #         cbar.set_label(r'$\Delta T(K)$')
#         #         plt.savefig(f"{outputfolder}\{self.modelname}_solution.png", dpi=300)

#         #         plt.clf()

#         #         ax, cbar = show(self.mesh, data=u, xlabel="x[mm]", ylabel="y[mm]", nLevs=10, cmap="Spectral_r", showMesh=True)# levels=np.linspace(min(u), max(u), 14), cMap='viridis', colorBar=True, label='Solution $u$')

#         #         ax.set_xlabel("x [mm]")
#         #         ax.set_ylabel("x [mm]")
#         #         ax.set_xlim([19.0,21.0])
#         #         ax.set_ylim([4, 4.5])

#         #         cbar.set_label(r'$\Delta T(K)$')

#         #         #ax.set_ylim([0, 150])

#         #         plt.savefig(f"{outputfolder}\{self.modelname}_withmesh_zoom_solution.png", dpi=300, bbox_inches='tight')

#         #         plt.clf()

#         #         ax, cbar = show(self.mesh, data=u, xlabel="x[mm]", ylabel="y[mm]", nLevs=10, cmap="Spectral_r", showMesh=True)# levels=np.linspace(min(u), max(u), 14), cMap='viridis', colorBar=True, label='Solution $u$')

#         #         ax.set_xlabel("x [mm]")
#         #         ax.set_ylabel("x [mm]")
#         #         ax.set_xlim([58.0,61.0])
#         #         ax.set_ylim([4, 4.5])

#         #         cbar.set_label(r'$\Delta T(K)$')

#         #         #ax.set_ylim([0, 150])

#         #         plt.savefig(f"{outputfolder}\{self.modelname}_withmesh_zoom2_solution.png", dpi=300, bbox_inches='tight')


        
#         # def __del__(self):
#         #         plt.close('all')

if __name__ == "__main__":

        scenarios = [
        ]

        # fthicknesses = np.linspace(3, 5, 11)
        # cfthicknesses = np.linspace(0.05, 0.2, 4)
        # tiODs = np.linspace(1.0, 3.0, 16)
        # tiWT = 0.15
        # gthickness = 0.1

        fthicknesses = np.linspace(3, 5, 5)
        cfthicknesses = np.linspace(0.05, 0.2, 3)
        tiODs = np.linspace(1.0, 3.0, 6)
        tiWT = 0.15
        gthickness = 0.1


        for fthickness in fthicknesses:
                for cfthickness in cfthicknesses:
                        for tiOD in tiODs:
                                if fthickness - tiOD - 2*gthickness < 0.5: continue
                                scenarios.append({
                                        "modelname": f"Cross_Optimizer_f{fthickness:.2f}_cf{cfthickness:.2f}_tiOD{tiOD:.2f}",
                                        "fthickness": fthickness,
                                        "cfthickness": cfthickness,
                                        "tlocy": fthickness/2 + cfthickness,
                                        "powerdensity": 0.4e-2,
                                        "tiOD": tiOD,
                                        "saveplots": True
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
                except:
                        nfailed += 1
                        print(f"Error in scenario: {scenario['modelname']}")
                        continue

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