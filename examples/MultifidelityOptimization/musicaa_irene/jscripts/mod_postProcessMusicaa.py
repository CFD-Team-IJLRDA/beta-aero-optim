import sys
import numpy                    as np 
import matplotlib.pyplot        as plt 
import matplotlib.font_manager  as font_manager
from mod_calcBoundaryLayer  import calculateBL
from matplotlib.animation   import FuncAnimation


# Class to Post-Process results form Musicaa, to plot data, 

class postProcess(calculateBL):
    """Class to postProcess results from Musicaa, this class inherits readResults and calulate BL
    The class here only plots data and animations"""
    def __init__(self, 
                 rep,                   # Repository from where to read data
                 ngh,                   # Number of ghost points
                 is_extended,           # Is the grid extended (with ghost points) or not
                 bloc,                  # Blocks with walls to compute wall values
                 repo_exp=None,         # Repository where experimental data are stored
                 bloc_fst=None,         # Blocks in the freestream of the BL to study turbulent intensity evolution
                 is_little_endian=True, # Is the format little endian or not
                 is_stats=True,         # PostProcess form stats if True, else from planes
                 is_sw_rv=False,        # Is it swap rv
                 is_curv_real=True,     # Is the mesh really curvilinear or is it enforced by Musicaa solver
                 gamma=1.4,             # Gas specific heat ratio, default value for diatomic perfect gas
                 r_gas=287.06,          # Specific gas constant, default value for air considered as a perfect gas
                 is_2D=True,            # Is the flow 2D or not
                 is_RANS=False,         # Is it a RANS simulation or not
                 model_RANS=None        # If it is RANS which model is used
                 ) -> None:
        super().__init__(rep, 
                         ngh, 
                         is_extended, 
                         bloc, bloc_fst, 
                         is_little_endian, 
                         is_stats, 
                         is_sw_rv, 
                         gamma, 
                         r_gas, 
                         is_2D, 
                         is_RANS, 
                         model_RANS)
        self.dict_t3a_1Wire     = {}
        self.dict_t3a_UVWire    = {}
        self.dict_t3a_UWWire    = {}
        self.mean_data_dict     = {}

        pass 
    
    def init_postProcess(self):
        """
        Initialize the post-processing tasks by reading necessary data.

        This method reads information about the simulation, including time steps,
        grid parameters, and other required data for plotting or analysis.

        Raises:
            FileNotFoundError: If the required simulation data files are not found.
            IOError: If there is an issue reading data from files.
            ValueError: If the data read from files is invalid or incomplete.
        """
        try:
            # Read simulation information
            self.read_info()
            self.read_time()
            self.read_param_blocks()
            self.read_grid(verbose=True, 
                           is_sw_rv=self.is_sw_rv)
            self.snap_type_characteristics()
            if self.is_stats: 
                self.read_stats_stbl(bloc=self.bloc_fst)
        except FileNotFoundError:
            raise FileNotFoundError("Simulation data files not found. Please ensure that all required files exist.")
        except IOError as e:
            raise IOError(f"Error reading data files: {e}.")
        except ValueError as e:
            raise ValueError(f"Invalid data encountered: {e}. Please check the simulation data files.")
        # Compute number of points in block to get BL data from
        self._nx_tot = sum(self.nx[which_bloc] for which_bloc in self.bloc)

    #=======================
    # Read experimental data
    #=======================
    def read_summary_file_to_dict(self, filename):
        """
        Reads a CASEy.txt file and stores its content in a dictionary.
        Usually the file contains: 
                    RUN       SINGLE WIRE TEST DATA RUN NUMBER
                    X         AXIAL POSITION (MM)
                    REX       REYNOLDS NUMBER BASED ON X AND UO
                    UO        FREESTREAM VELOCITY (M/S)
                    CF        SKIN FRICTION COEFFICIENT
                    H         SHAPE FACTOR
                    RETHETA   REYNOLDS NUMBER BASED ON MOMENTUM THICKNESS
                    DELTA     99% BOUNDARY LAYER THICKNESS (MM)
                    TUFS      FREESTREAM TURBULENCE INTENSITY (%)
        Args:            
            filename: The path to the summary.txt file.
        return: 
            A dictionary with the file's content.
        """
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()

                # Find the line with column headers (assuming it's the one that contains 'RUN' keyword)
                for i, line in enumerate(lines):
                    if 'RUN' in line and not "KEY" in line:
                        headers_line = i
                        break
                    
                # Extract headers
                headers = lines[headers_line].split()
                for header in headers:
                    self.mean_data_dict[header] = []

                # Extract data rows
                for line in lines[headers_line+1:]:
                    # Strip any leading/trailing whitespace characters including newlines
                    line = line.strip()

                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Split the line into data entries
                    data_entries = line.split()

                    # Append each data entry to the respective header list
                    for header, entry in zip(headers, data_entries):
                        if header != "RUN":
                            self.mean_data_dict[header].append(float(entry))
                        else:
                            self.mean_data_dict[header].append(entry)

        except FileNotFoundError:
            print(f"The file {filename} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")
        return self.mean_data_dict

    ### Read advanced T3A data:
    def read_T3A_data(self, rep_t3a):
        print("\nReading T3A data...")
        # Reading single wire data
        try:
            f=open(rep_t3a+'T3A_single_wire.txt','r')
            print("~> Reading T3A_single_wire.txt")
            ind = 0
        except FileNotFoundError:
            print("~> File T3A_single_wire.txt not found...")
            self.dict_t3a_1Wire,ind = None,-1
        if ind==0:
            lines = f.readlines()
            line = lines[0].split()
            while True:
                while line[0]!='RUN':
                    ind += 1;line = lines[ind].split()
                    if len(line)==0: line=[0]
                name_run = line[2][3:]
                self.dict_t3a_1Wire[name_run] = {'Y':[],'Y/DEL':[],'Y+':[],'U':[],'U/UO':[],'U+':[],'u':[],'u/UO':[],'u/U':[],'u/UTAU':[]}
                ind += 1;line = lines[ind].split()
                x_p = float(line[3])
                self.dict_t3a_1Wire[name_run]['x'] = x_p
                while line[1]!='VELOCITY':
                    ind += 1;line = lines[ind].split()
                self.dict_t3a_1Wire[name_run]['U_tau'] = float(line[8])
                ind += 1;line = lines[ind].split()
                numb = float(line[8])
                numb2 = int(line[9][-1])
                self.dict_t3a_1Wire[name_run]['Rex'] = numb*10**numb2
                ind += 1;line = lines[ind].split()
                self.dict_t3a_1Wire[name_run]['nu'] = float(line[8])*10**-5
                while line[0]!='MM':
                    ind += 1;line = lines[ind].split()
                    if len(line)==0: line=[0]
                ind += 1;line = lines[ind].split()
                while len(line)>0:
                    self.dict_t3a_1Wire[name_run]['Y'].append(float(line[0]))
                    self.dict_t3a_1Wire[name_run]['Y/DEL'].append(float(line[1]))
                    self.dict_t3a_1Wire[name_run]['Y+'].append(float(line[2]))
                    self.dict_t3a_1Wire[name_run]['U'].append(float(line[3]))
                    self.dict_t3a_1Wire[name_run]['U/UO'].append(float(line[4]))
                    self.dict_t3a_1Wire[name_run]['U+'].append(float(line[5]))
                    self.dict_t3a_1Wire[name_run]['u'].append(float(line[6]))
                    self.dict_t3a_1Wire[name_run]['u/UO'].append(float(line[7]))
                    self.dict_t3a_1Wire[name_run]['u/U'].append(float(line[8]))
                    self.dict_t3a_1Wire[name_run]['u/UTAU'].append(float(line[9]))
                    try:
                        ind += 1;line = lines[ind].split()
                    except IndexError:
                        break
                try:
                    ind += 1;line = lines[ind].split()
                except IndexError:
                    break

        # Reading UV cross wire data
        try:
            f=open(rep_t3a+'T3A_UV_cross_wire.txt','r')
            print("~> Reading T3A_UV_cross_wire.txt")
            ind = 0
        except FileNotFoundError:
            print("~> File T3A_UV_cross_wire.txt not found...")
            self.dict_t3a_UVWire,ind = None,-1
        if ind==0:
            lines = f.readlines()
            line = lines[0].split()
            while True:
                while line[0]!='RUN':
                    ind += 1;line = lines[ind].split()
                    if len(line)==0: line=[0]
                name_run = line[2][3:]
                self.dict_t3a_UVWire[name_run] = {'Y':[],'U':[],'V':[],'u':[],'v':[],'uv':[],'u/v':[],'u/U':[],'v/U':[],'uv/u.v':[]}
                ind += 3;line = lines[ind].split()
                while len(line)>0:
                    self.dict_t3a_UVWire[name_run]['Y'].append(float(line[0]))
                    self.dict_t3a_UVWire[name_run]['U'].append(float(line[1]))
                    self.dict_t3a_UVWire[name_run]['V'].append(float(line[2]))
                    self.dict_t3a_UVWire[name_run]['u'].append(float(line[3]))
                    self.dict_t3a_UVWire[name_run]['v'].append(float(line[4]))
                    self.dict_t3a_UVWire[name_run]['uv'].append(float(line[5]))
                    self.dict_t3a_UVWire[name_run]['u/v'].append(float(line[6]))
                    self.dict_t3a_UVWire[name_run]['u/U'].append(float(line[7]))
                    self.dict_t3a_UVWire[name_run]['v/U'].append(float(line[8]))
                    self.dict_t3a_UVWire[name_run]['uv/u.v'].append(float(line[9]))
                    try:
                        ind += 1;line = lines[ind].split()
                    except IndexError:
                        break
                try:
                    ind += 1;line = lines[ind].split()
                except IndexError:
                    break

        # Reading UW cross wire data
        try:
            f=open(rep_t3a+'T3A_UW_cross_wire.txt','r')
            print("~> Reading T3A_UW_cross_wire.txt")
            ind = 0
        except FileNotFoundError:
            print("~> File T3A_UW_cross_wire.txt not found...")
            ind = -1
        if ind==0:
            lines = f.readlines()
            line = lines[0].split()
            while True:
                while line[0]!='RUN':
                    ind += 1;line = lines[ind].split()
                    if len(line)==0: line=[0]
                name_run = line[2][3:]
                self.dict_t3a_UWWire[name_run] = {'Y':[],'U':[],'W':[],'u':[],'w':[],'uw':[],'u/w':[],'u/U':[],'w/U':[],'uw/u.w':[]}
                ind += 3;line = lines[ind].split()
                while len(line)>0:
                    self.dict_t3a_UWWire[name_run]['Y'].append(float(line[0]))
                    self.dict_t3a_UWWire[name_run]['U'].append(float(line[1]))
                    self.dict_t3a_UWWire[name_run]['W'].append(float(line[2]))
                    self.dict_t3a_UWWire[name_run]['u'].append(float(line[3]))
                    self.dict_t3a_UWWire[name_run]['w'].append(float(line[4]))
                    self.dict_t3a_UWWire[name_run]['uw'].append(float(line[5]))
                    self.dict_t3a_UWWire[name_run]['u/w'].append(float(line[6]))
                    self.dict_t3a_UWWire[name_run]['u/U'].append(float(line[7]))
                    self.dict_t3a_UWWire[name_run]['w/U'].append(float(line[8]))
                    self.dict_t3a_UWWire[name_run]['uw/u.w'].append(float(line[9]))
                    try:
                        ind += 1;line = lines[ind].split()
                    except IndexError:
                        break
                try:
                    ind += 1;line = lines[ind].split()
                except IndexError:
                    break


    #==============================
    # Plots and flow vizualization:
    #==============================
    def plot_grid(self, save_path, name, is_visual=False):
        """
        Plot the grid with a different color for each block if it is a multi-block domain.

        Args:
            save_path (str): Path to save the plot.
            name (str): Name of the grid.
            is_visual (bool, optional): Option to show the plot interactively. Default is False.

        Returns:
            None
        """
        plt.figure(figsize=(15, 9))
        for block_index in range(1, self.dict_info["nbloc"] + 1):
            plt.title(f'Grid for {name}')
            if self.dict_info["is_curv"] == 'T':
                for i in range(self.nx[block_index]):
                    plt.plot(self.x[block_index][i, :], self.y[block_index][i, :], color='black', linewidth=0.5)
                    plt.plot(self.x[block_index][-1, :], self.y[block_index][-1, :], color='black', linewidth=0.5)
                for j in range(self.ny[block_index]):
                    plt.plot(self.x[block_index][:, j], self.y[block_index][:, j], color='black', linewidth=0.5)
                    plt.plot(self.x[block_index][:, -1], self.y[block_index][:, -1], color='black', linewidth=0.5)
            else:
                for i in range(self.nx[block_index]):
                    plt.plot([self.x[block_index][i]] * self.ny[block_index], self.y[block_index][:], color='black', linewidth=0.5)
                    plt.plot([self.x[block_index][-1]] * self.ny[block_index], self.y[block_index][:], color='black', linewidth=0.5)
                for j in range(self.ny[block_index]):
                    plt.plot(self.x[block_index][:], [self.y[block_index][j]] * self.nx[block_index], color='black', linewidth=0.5)
                    plt.plot(self.x[block_index][:], [self.y[block_index][-1]] * self.nx[block_index], color='black', linewidth=0.5)
        plt.axis("equal")
        plt.savefig(save_path)
        if is_visual:
            plt.show()
        
    def update_frame_xy(self, 
                        frame, 
                        ax, 
                        varname, 
                        norm,
                        levels, 
                        plane_nb,
                        x_lab="X-axis",
                        y_lab="Y-axis",
                        title="data",
                        ref=1.0, 
                        colormap="Spectral_r"):
        
        # Clear previous contour plots
        for coll in ax.collections:
            coll.remove()
    
        # Update contour plots with new frame data
        for i in range(1, self.dict_info["nbloc"] + 1):
            ax.contourf(self.x[i], self.y[i], self.planes[i][plane_nb][varname][frame]/ref, norm=norm, levels=levels, cmap='Spectral_r')
        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)
        ax.axis("equal")
        ax.set_title(rf"{title}")

    def update_frame_xz(self, 
                        frame, 
                        ax, 
                        varname, 
                        norm,
                        levels, 
                        plane_nb,
                        x_lab="X-axis",
                        y_lab="Y-axis",
                        title="data",
                        ref=1.0, 
                        colormap="Spectral_r"):
        
        # Clear previous contour plots
        for coll in ax.collections:
            coll.remove()
    
        # Update contour plots with new frame data
        for i in range(1, self.dict_info["nbloc"] + 1):
            ax.contourf(self.x[i][:,0], self.z[i], self.planes[i][plane_nb][varname][frame].transpose()/ref, norm=norm, levels=levels, cmap='Spectral_r')
        ax.set_xlabel(x_lab, fontsize=14)
        ax.set_ylabel(y_lab, fontsize=14)
        ax.axis("equal")
        ax.set_title(rf"{title}", fontsize=16)

    # ================
    # Get wall values:
    # ================
    def calc_bl(self, 
                normal_w=None, 
                tangent_w=None, 
                is_turb=False,
                jmin=35, 
                jmax=45):
        """
        Plots basic values: Skin friction coefficient, Turbulent intensity, yplus and xplus along x axis,
                            boundary layer thicknesses and

    
        Args:
            normal_w (array, optional): Normal vector of the wall. Defaults to None.
            tangent_w (array, optional): Tangent vector of the wall. Defaults to None.
            is_turb (boolean, optional): is it turbulent or not
            jmin_pos (int, optional): Minimum vertical index for turbulent intensity calculation.
            jmax_pos (int, optional): Maximum vertical index for turbulent intensity calculation. 
    
        Returns:
            tuple: Tuple containing numpy arrays of boundary layer values.
        Raises:
            ValueError: If the normal to the surface is not provided
            ValueError: If the tangent to the surface is not provided for curvilinear meshes
        """
        # Error handling for missing parameters
        if normal_w is None:
            raise ValueError("normal_w must be provided.")
        # Offset: 
        offset = 0

        # Get Boundary Layer thickenss:
        self.calc_d99_thickness(normal_w=normal_w)
        self.calc_displacement_thickness()
        self.calc_momentum_thickness()
        # Declare y_plus and x_plus arrays: 
        x_plus_x = np.zeros(self._nx_tot-1)
        y_plus_x = np.zeros(self._nx_tot)
        y_delta = np.zeros(self._nx_tot)
        y_1 = np.zeros(self._nx_tot)
        y_0 = np.zeros(self._nx_tot)
        # Turbulent intensity: 
        if is_turb: Tu_sim, tu_x, tu_y, tu_z = self.calc_turb_intensity(jmax_pos=jmax, jmin_pos=jmin, )

        dudn_tot=[]
        
        for block_index in self.bloc:
            # Calculate BL values
    
            if self.dict_info["is_curv"]=="T":
                if tangent_w is None:
                    raise ValueError("Both normal_w and tangent_w must be provided for curvilinear meshes.")
                #print("the size of nx is", self.nx)
                # Calculate dudn for curved mesh
                dudn =  (
                        (
                        -tangent_w[1,offset:offset+self.nx[block_index]] * self._stats2[block_index]["rho*duy"][:,1] / self._stats1[block_index]["rho"][:,1] 
                        +tangent_w[0,offset:offset+self.nx[block_index]] * self._stats2[block_index]["rho*duy"][:,1] / self._stats1[block_index]["rho"][:,1]
                        ) * tangent_w[0,offset:offset+self.nx[block_index]]
                        +
                        (
                        -tangent_w[1,offset:offset+self.nx[block_index]] * self._stats2[block_index]["rho*dvx"][:,1] / self._stats1[block_index]["rho"][:,1] 
                        +tangent_w[0,offset:offset+self.nx[block_index]] * self._stats2[block_index]["rho*dvy"][:,1] / self._stats1[block_index]["rho"][:,1]
                        ) * tangent_w[1,offset:offset+self.nx[block_index]]
                        )
                dudn_tot.extend(dudn)
                
                #print("dudn tot is ", dudn_tot)
        

        dudn_tot = np.array(dudn_tot)

        y_plus_x_real = {}

        for block_index in self.bloc:
            # Calculate BL values
    
            if self.dict_info["is_curv"]=="T":
                if tangent_w is None:
                    raise ValueError("Both normal_w and tangent_w must be provided for curvilinear meshes.")
                #print("the size of nx is", self.nx)
                # Calculate dudn for curved mesh
                dudn =  (
                        (
                        -tangent_w[1,offset:offset+self.nx[block_index]] * self._stats2[block_index]["rho*duy"][:,1] / self._stats1[block_index]["rho"][:,1] 
                        +tangent_w[0,offset:offset+self.nx[block_index]] * self._stats2[block_index]["rho*duy"][:,1] / self._stats1[block_index]["rho"][:,1]
                        ) * tangent_w[0,offset:offset+self.nx[block_index]]
                        +
                        (
                        -tangent_w[1,offset:offset+self.nx[block_index]] * self._stats2[block_index]["rho*dvx"][:,1] / self._stats1[block_index]["rho"][:,1] 
                        +tangent_w[0,offset:offset+self.nx[block_index]] * self._stats2[block_index]["rho*dvy"][:,1] / self._stats1[block_index]["rho"][:,1]
                        ) * tangent_w[1,offset:offset+self.nx[block_index]]
                        )
                #print("we are good good!")
                #print("length of dudn is", len(dudn))
                #print("self nx for block", block_index, "is", self.nx[block_index])
                #print("self nx tot is", self._nx_tot)

                # Get wall values:
                #tau_w, utau, Cf, Re_tau = self.calc_utau_cf(duTdn=dudn)
                tau_w, utau, Cf, Re_tau = self.calc_utau_cf(duTdn=dudn_tot)
                # Compute yplus and xplus:
                #print("y in block one is", self.y[1])
                #print("size of utau is", utau.size)
                #print("Roref is", self.dict_info['Roref'])
                #print("Muref is", self.dict_info["Muref"])
                #y_plus_x[offset:offset+self.nx[block_index]] = utau * (self.y[block_index][:,1] - self.y[block_index][:,0]) \
                #                                                    * self.dict_info["Roref"] / self.dict_info["Muref"]
                #x_plus_x[offset:offset+self.nx[block_index]] = 0.5 * (utau[1:] + utau[:-1]) * (self.x[block_index][1:,1] - self.x[block_index][:-1,1]) \
                #                                                                            * self.dict_info["Roref"] / self.dict_info["Muref"]
                #y_plus_x[offset:offset+self.nx[block_index]] = utau[offset:offset+self.nx[block_index]] * (self.y[block_index][:,1] - self.y[block_index][:,0]) \
                #                                                    * self.dict_info["Roref"] / self.dict_info["Muref"]
                y_plus_x[offset:offset+self.nx[block_index]] = utau[offset:offset+self.nx[block_index]] * np.sqrt(((self.y[block_index][:,1] - self.y[block_index][:,0])**2 + (self.x[block_index][:,1] - self.x[block_index][:,0])**2))/1000 \
                                                                    * self.dict_info["Roref"] / self.dict_info["Muref"]
                #x_plus_x[offset:offset+self.nx[block_index]] = 0.5 * (utau[offset:offset+self.nx[block_index]][1:] + utau[offset:offset+self.nx[block_index]][:-1]) * (self.x[block_index][1:,1] - self.x[block_index][:-1,1]) \
                #                                                                            * self.dict_info["Roref"] / self.dict_info["Muref"]
                y_delta[offset:offset+self.nx[block_index]] = (abs(self.y[block_index][:,1] - self.y[block_index][:,0]))/1000
                y_1[offset:offset+self.nx[block_index]] = (self.y[block_index][:,49])/1000
                y_0[offset:offset+self.nx[block_index]] = (self.y[block_index][:,0])/1000

                #print(self.y[block_index][:,1]-self.y[block_index][:,0])
                #print(self.bloc)
                #print(abs(utau[offset:offset+self.nx[block_index]]) * np.sqrt((self.y[block_index][:,1] - self.y[block_index][:,0])**2 + (self.x[block_index][:,1] - self.x[block_index][:,0])**2)/1000 \
                #                                                    * self.dict_info["Roref"] / self.dict_info["Muref"])

                y_plus_x_real[block_index] = abs(utau[offset:offset+self.nx[block_index]]) * np.sqrt((self.y[block_index][:,1] - self.y[block_index][:,0])**2 + (self.x[block_index][:,1] - self.x[block_index][:,0])**2)/1000 \
                                                                    * self.dict_info["Roref"] / self.dict_info["Muref"]
                
                
                #print(y_plus_x_real)
                
            else:
                # Calculate BL values for non-curved mesh
                tau_w, utau, Cf, Re_tau = self.calc_utau_cf(data=self._stats1[block_index])

                y_plus_x[offset:offset+self.nx[block_index]] = utau * (self.y[block_index][1]-self.y[block_index][0]) \
                                                                    * self.dict_info["Roref"] / self.dict_info["Muref"]
                x_plus_x[offset:offset+self.nx[block_index]] = 0.5  * (utau[1:] + utau[:-1]) * (self.x[block_index][1:] - self.x[block_index][:-1]) \
                                                                    * self.dict_info["Roref"] / self.dict_info["Muref"]

        if not is_turb:
            return tau_w, utau, Cf, Re_tau, y_plus_x, x_plus_x
        else:
            return tau_w, utau, Cf, Re_tau, y_plus_x, x_plus_x, Tu_sim, tu_x, tu_y, tu_z, y_delta, y_1, y_0, y_plus_x_real

    def wall_normal_profiles(self, Re_th, Ue=None):
        """Get wall normal velocity profiles to compare with experiments"""
        print("In wall normal profiles")
        # Determination of the i indices of the profiles to show for Re_x
        i2plot,key2plot = {},{}
        i2plot_t,key2plot_t = {},{}
        key2plot2       = {}
        key2plot2_t     = {}
        for index in self.bloc:
            Re_x = self.return_reynolds_x(x_in=0.0, Ue=Ue, bloc=index)
            i2plot[index], key2plot[index]  = [], []
            for key in self.dict_t3a_1Wire.keys():
                itemp = 0
                while Re_x[itemp]<self.dict_t3a_1Wire[key]['Rex']:
                    itemp+=1
                    if itemp==self.nx[index]-1: break
                if itemp!=self.nx[index]-1:
                    i2plot[index].append(itemp)
                    key2plot[index].append(key)
            key2plot2[index] = []
            for key in key2plot:
                if key in self.dict_t3a_UVWire.keys(): key2plot2[index].append(key)
        
            # Creation of dict_t3a_1Wire[key]['Re_th']
            for key in key2plot[index]:
                i = 0
                while np.round(self.mean_data_dict["REX"][i])!=np.round(self.dict_t3a_1Wire[key]['Rex']) and i<len(self.mean_data_dict["REX"]):
                    i+=1
                self.dict_t3a_1Wire[key]['Re_th'] = self.mean_data_dict["RETHETA"][i]
        # Determination of the i indices of the profiles to show for Re_theta
        i2plot_t[index],key2plot_t[index] = [],[]
        for key in self.dict_t3a_1Wire.keys():
            itemp = 0
            if 'Re_th' in self.dict_t3a_1Wire[key].keys():
                while Re_th[itemp]<self.dict_t3a_1Wire[key]['Re_th']:
                    itemp+=1
                    if itemp==self.nx[index]-50: break
                if itemp!=self.nx[index]-50:
                    i2plot_t[index].append(itemp)
                    key2plot_t[index].append(key)
        key2plot2_t[index] = []
        for key in key2plot_t[index]:
            if key in self.dict_t3a_UVWire.keys(): key2plot2_t[index].append(key)
        
        return key2plot, key2plot_t, i2plot, i2plot_t, key2plot2, key2plot2_t
