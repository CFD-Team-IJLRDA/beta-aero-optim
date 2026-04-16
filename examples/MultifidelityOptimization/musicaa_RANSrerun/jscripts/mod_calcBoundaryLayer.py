import logging
import sys 
import numpy        as np 
from mod_readData   import readResultsMusicaa

###### Define postProcessBL class:

class calculateBL(readResultsMusicaa):
    """Class to postProcess Boundary Layer flows in Musicaa.

    This class inherits from readResultsMusicaa and provides additional functionality
    for post-processing boundary layer flows. It supports processing data from both
    stats and plane files.

    Additional features:
        - bloc: List of blocks with walls to compute wall values.
        - bloc_fst: Blocks in the freestream of the boundary layer to study turbulent intensity evolution.
        - is_stats: Boolean indicating whether to post-process from stats files. If False, post-process from plane files.
        - is_extended: Boolean indicating whether the flow is extended or 3D.
        - is_sw_rv: Boolean indicating whether to swap rv.
        - is_curv_real: Boolean indicating whether the mesh is really curvilinear or enforced by Musicaa solver.
    """

    def __init__(self, 
                 rep,
                 ngh,
                 is_extended,                       
                 bloc,                      
                 bloc_fst=None,             
                 is_little_endian=True,     
                 is_stats=True,             
                 is_sw_rv=False,            
                 gamma=1.4,                 
                 r_gas=287.06,              
                 is_2D=True,                
                 is_RANS=False,             
                 model_RANS=None           
                 ) -> None:
        """Initialize the calculateBL object."""
        super().__init__(rep, ngh, gamma, is_extended, r_gas, is_little_endian, is_2D, is_RANS, model_RANS)
        
        # Validate and store bloc and bloc_fst as lists
        self.bloc = [bloc] if isinstance(bloc, int) else bloc
        self.bloc_fst = [bloc_fst] if isinstance(bloc_fst, int) else bloc_fst
        # Validate other attributes:
        self.is_stats = is_stats
        self.is_extended = is_extended
        self.is_sw_rv = is_sw_rv

        # Error handling for invalid bloc or bloc_fst values
        if not self.bloc:
            raise ValueError("At least one block must be provided.")
        self.read_info()
        if self.bloc_fst and any(bloc > self.dict_info["nbloc"] for bloc in self.bloc_fst):
            raise ValueError("Invalid bloc_fst value. Must be within range of block indices.")

    def init_calc_bl(self, read_time=True, read_param_blocks=True, read_grid=True):
        """Initialize calculation for boundary layer post-processing.

        This function performs several initialization steps:
        1. Read simulation information.
        2. Read simulation time data.
        3. Read parameter blocks.
        4. Read grid data.

        Parameters:
        - read_info: Whether to read simulation information. Default is True.
        - read_time: Whether to read simulation time data. Default is True.
        - read_param_blocks: Whether to read parameter blocks. Default is True.
        - read_grid: Whether to read grid data. Default is True.
        """
        
        if read_time:
            logging.info("Reading simulation time data...")
            try:
                self.read_time()
                logging.info("Simulation time data read successfully.")
            except Exception as e:
                logging.error(f"Error reading simulation time data: {e}")
        
        if read_param_blocks:
            logging.info("Reading parameter blocks...")
            try:
                self.read_param_blocks()
                logging.info("Parameter blocks read successfully.")
            except Exception as e:
                logging.error(f"Error reading parameter blocks: {e}")
        
        if read_grid:
            logging.info("Reading grid data...")
            try:
                self.snap_type_characteristics()
                self.read_grid(verbose=True,
                               is_extended=self.is_extended, 
                               is_sw_rv=self.is_sw_rv)
                logging.info("Grid data read successfully.")
            except Exception as e:
                logging.error(f"Error reading grid data: {e}")
        # Compute number of points in block to get BL data from
        self._nx_tot = sum(self.nx[which_bloc] for which_bloc in self.bloc)
        

    def return_reynolds_x(self, x_in, Ue=None, bloc=None, verbose=False):
        """Returns the Reynolds number along the plate and the inlet Reynolds number for each block in bloc."""
        Re_x = []

        if not isinstance(Ue, np.ndarray): Ue= self.dict_info["Uref"]
        if bloc==None: 
            bloc = self.bloc
        else:
            bloc = [bloc] if isinstance(bloc, int) else bloc

        for which_bloc in bloc:
            x_block = self.x[which_bloc]
            nx_block = self.nx[which_bloc]

            for i in range(nx_block):
                if self.dict_info["is_curv"]=="T" and self.is_2D:
                    x_coord = x_block[i, 0] + x_in
                elif self.dict_info["is_curv"]=="T" and not self.is_2D:
                    x_coord = x_block[i, 0, 0] + x_in
                else:
                    x_coord = x_block[i] + x_in

                Re = x_coord * self.dict_info["Roref"] * Ue / self.dict_info["Muref"]
                Re_x.append(Re)
            if verbose:
                print("Length of domain for case %s in block %i: %.2e < Re_x < %.2e" % (self.repo, which_bloc, Re_x[0], Re_x[-1]))
                if self.dict_info["is_curv"]=="T" and self.is_2D:
                    print("     ~> x=%.2e: Re_x=%.2e" % (x_block[:, 0][nx_block // 4], Re_x[nx_block // 4]))
                    print("     ~> x=%.2e: Re_x=%.2e" % (x_block[:, 0][nx_block // 2], Re_x[nx_block // 2]))
                    print("     ~> x=%.2e: Re_x=%.2e" % (x_block[:, 0][nx_block * 3 // 4], Re_x[nx_block * 3 // 4]))
                else:
                    print("     ~> x=%.2e: Re_x=%.2e" % (x_block[:][nx_block // 4], Re_x[nx_block // 4]))
                    print("     ~> x=%.2e: Re_x=%.2e" % (x_block[:][nx_block // 2], Re_x[nx_block // 2]))
                    print("     ~> x=%.2e: Re_x=%.2e" % (x_block[:][nx_block * 3 // 4], Re_x[nx_block * 3 // 4]))

        return np.array(Re_x)

    # ======================
    # 99% thickness
    # ======================

    def calc_d99_thickness(self, U0=None, L=1, normal_w=None):
        """
        Calculates the 99% boundary layer thickness (delta_99) given the velocity field U0.

        Parameters:
            U0 (ndarray): Velocity field array. If None, it is copied from stats.
            L (float): Length scale of the flow (default is 1).
            normal_w (ndarray, optional): Array containing the normal direction components 
                                          for curvilinear meshes. Required if is_curv_real is True.

        Returns:
            ndarray, ndarray: Arrays containing the delta_99 thickness values and 
                              corresponding jmax indices for each block in bloc.

        Raises:
            ValueError: If U0 is None or if normal_w is required but not provided.

        Notes:
            The function calculates the delta_99 boundary layer thickness for each block 
            specified in bloc. If the mesh is not curvilinear, the function computes delta_99 
            using the nearest wall coordinate and velocity profile. If the mesh is curvilinear, 
            the normal_w array must be provided to compute the distance along the normal direction.
        """

        Ue = [self.dict_info["Mref"] * self.dict_info["cref"] for _ in range(self._nx_tot)]
        self._d99 = np.zeros(self._nx_tot)  # Initialize array to store d99 values for all blocks
        self._jmax = np.zeros(self._nx_tot, dtype='int')  # Initialize array to store jmax values for all blocks
        offset = 0  # Offset to keep track of current position in arrays

        for which_bloc in self.bloc:
            if self.is_stats:
                U0 = np.array(self._stats1[which_bloc]["u"])
            if U0 is None:
                raise ValueError("U0 must be provided or read form stats.")

            d99_block = np.zeros(self.nx[which_bloc])
            jmax_block = np.zeros(self.nx[which_bloc], dtype='int')

            for i in range(self.nx[which_bloc]):
                j = 0
                while U0[i, j] < 0.99 * Ue[i] and j < self.ny[which_bloc] - 1:
                    j += 1

                if self.dict_info["is_curv"] == "F":
                    d99_block[i] = ((self.y[which_bloc][j] - self.y[which_bloc][j - 1]) / 
                                    (U0[i, j] - U0[i, j - 1]) * (0.99 * Ue[i] - U0[i, j - 1]) + 
                                    self.y[which_bloc][j - 1]) * L
                else:
                    if normal_w is None:
                        raise ValueError("normal_w must be provided for curvilinear meshes.")

                    l_jm1 = ((self.y[which_bloc][i, j - 1] - self.y[which_bloc][i, 0]) ** 2 + 
                             (self.x[which_bloc][i, j - 1] - self.x[which_bloc][i, 0]) ** 2) ** 0.5
                    dl_j = ((self.y[which_bloc][i, j] - self.y[which_bloc][i, j - 1]) ** 2 + 
                            (self.x[which_bloc][i, j] - self.x[which_bloc][i, j - 1]) ** 2) ** 0.5
                    l_j = dl_j * (0.99 * Ue[i] - U0[i, j - 1]) / (U0[i, j] - U0[i, j - 1]) + l_jm1
                    d99_block[i] = np.abs(l_j * normal_w[1, i])

                jmax_block[i] = j

            self._d99[offset:offset+self.nx[which_bloc]] = d99_block  # Assign d99 values for this block to overall array
            self._jmax[offset:offset+self.nx[which_bloc]] = jmax_block  # Assign jmax values for this block to overall array
            offset += self.nx[which_bloc]  # Update offset for next block

        return self._d99, self._jmax
   
    # ======================
    # Displacement thickness
    # ======================

    def calc_displacement_thickness(self, 
                                U0=None,
                                rho=None,
                                is_compressible=False):
        """
        Calculates the displacement thickness (delta*) given the velocity field U0.

        Parameters:
            U0 (ndarray): Velocity field array. If None, it must be provided externally.
            rho (ndarray, optional): Density field array. Required for compressible flows.
            is_compressible (bool): Indicates if the flow is compressible or not (default is False).

        Returns:
            ndarray: Array containing the displacement thickness values for each block in bloc.

        Raises:
            SystemExit: If the density field (rho) is not provided for compressible flows.

        Notes:
            The function calculates the displacement thickness for each block specified in bloc. 
            If the flow is not compressible, it uses the velocity field U0 to compute delta*. 
            If the flow is compressible, both the velocity field U0 and the density field rho 
            must be provided.
        """
        self._deltas = np.zeros(self._nx_tot)
        rho_e  = [self.dict_info["Roref"] for _ in range(self._nx_tot)]
        Ue     = [self.dict_info["Mref"] * self.dict_info["cref"] for _ in range(self._nx_tot)]
        offset = 0  # Offset to keep track of current position in arrays

        # Start calculation:
        for which_bloc in self.bloc:
            if self.is_stats: 
                #U0 = self._stats1[which_bloc]["u"]
                U0 = (self._stats1[which_bloc]["u"]**2 + self._stats1[which_bloc]["v"]**2)**0.5
                rho = self._stats1[which_bloc]["rho"]
            # Define local variable for specific block: 
            deltas_block    = np.zeros(self.nx[which_bloc])
            if self.dict_info["is_curv"] == "F":
                if not is_compressible:
                    for i in range(self.nx[which_bloc]):
                        for j in range(self._jmax[i]):
                            arg1            = 1 - U0[i, j] / Ue[i]
                            arg2            = 1 - U0[i, j + 1] / Ue[i]
                            deltas_block[i] += (arg1 + arg2) / 2 * (self.y[which_bloc][j + 1] - self.y[which_bloc][j])
                        
                else:
                    if rho is None:
                        raise SystemExit("The density field must be provided for compressible flows.")

                    for i in range(self.nx[which_bloc]):
                        for j in range(self._jmax[i]):
                            arg1            = 1 - rho[i, j] * U0[i, j] / rho_e[m] / Ue[i]
                            arg2            = 1 - rho[i, j + 1] * U0[i, j + 1] / rho_e[m] / Ue[i]
                            deltas_block[i] +=  self._deltas[i] + (arg1 + arg2) / 2 * (self.y[which_bloc][j + 1] - self.y[which_bloc][j])
            else:
                for i in range(self.nx[which_bloc]):
                    for j in range(self._jmax[i]):
                        arg1            = 1 - rho[i, j] * U0[i, j] / rho_e[i] / Ue[i]
                        arg2            = 1 - rho[i, j + 1] * U0[i, j + 1] / rho_e[i] / Ue[i]
                        #deltas_block[i] +=  self._deltas[i] + (arg1 + arg2) / 2 * ((self.x[which_bloc][i, j + 1] - self.x[which_bloc][i, j]) ** 2 \
                        #                                                        +  (self.y[which_bloc][i, j + 1] - self.y[which_bloc][i, j]) ** 2) ** 0.5
                        deltas_block[i] += (arg1 + arg2) / 2 * ((self.x[which_bloc][i, j + 1] - self.x[which_bloc][i, j]) ** 2 \
                                                                                +  (self.y[which_bloc][i, j + 1] - self.y[which_bloc][i, j]) ** 2) ** 0.5
            self._deltas[offset:offset+self.nx[which_bloc]] = deltas_block # Assign deltas value for this block to overall array
            offset += self.nx[which_bloc]  # Update offset for next block

        return self._deltas

    # ======================
    # Displacement thickness
    # ======================

    def calc_momentum_thickness(self, 
                            U0=None, 
                            rho=None, 
                            is_compressible=False):
        """
        Calculates the momentum thickness of the boundary layer.

        Parameters:
            U0 (ndarray): Velocity field array. If None, it must be provided externally.
            rho (ndarray, optional): Density field array. Required for compressible flows.
            is_compressible (bool): Indicates if the flow is compressible or not (default is False).

        Returns:
            ndarray: Array containing the momentum thickness values for each block in bloc.

        Raises:
            SystemExit: If the density field (rho) is not provided for compressible flows.

        Notes:
            The function calculates the momentum thickness for each block specified in bloc. 
            If the flow is compressible, both the velocity field U0 and the density field rho 
            must be provided.
        """
        self._deltatheta = np.zeros(self._nx_tot)  
        rho_e = self.dict_info["Roref"]
        Ue = [self.dict_info["Mref"] * self.dict_info["cref"] for _ in range(self._nx_tot)]
        offset = 0 # Offset to keep track of current position in arrays
        # Start calculation:
        #print(U0)
        #print(self.is_stats)
        for which_bloc in self.bloc:
            if self.is_stats and U0 is None: U0 = self._stats1[which_bloc]["u"]
            #if self.is_stats and U0==None: U0 = self._stats1[which_bloc]["u"]
            deltatheta = np.zeros(self.nx[which_bloc])
            if self.dict_info["is_curv"] == "F":
                if is_compressible:
                    if rho is None:
                        raise SystemExit("The density field must be provided for compressible flows.")
                    for i in range(self.nx[which_bloc]):
                        for j in range(self.jmax[i] + 10):
                            arg1            = rho[i, j] * U0[i, j] / rho_e[i] / Ue[i] * (1 - U0[i, j] / Ue[i])
                            arg2            = rho[i, j + 1] * U0[i, j + 1] / rho_e[i] / Ue[i] * (1 - U0[i, j + 1] / Ue[i])
                            deltatheta[i]   += (arg1 + arg2) / 2 * (self.y[self.bloc][j + 1] - self.y[self.bloc][j])
                else:
                    for i in range(self.nx[which_bloc]):
                        for j in range(min(len(self.y[which_bloc]) - 1, self._jmax[i] + 10)):
                            arg1            = U0[i, j] / Ue[i] * (1 - U0[i, j] / Ue[i])
                            arg2            = U0[i, j + 1] / Ue[i] * (1 - U0[i, j + 1] / Ue[i])
                            deltatheta[i]   += (arg1 + arg2) / 2 * (self.y[which_bloc][j + 1] - self.y[which_bloc][j])
            else:
                for i in range(2):
                #for i in range(self.nx[which_bloc]):
                    #print("this is the", i ,"th iteration for i")
                    for j in range(1):
                    #for j in range(min(self.y[which_bloc].shape[1] - 1, self._jmax[i] + 10)):
                        #print("this is the", j ,"th iteration for j")
                        #print("shape of U0 is", U0.shape)
                        arg1            = U0[i, j] / Ue[i] * (1 - U0[i, j] / Ue[i])
                        arg2            = U0[i, j + 1] / Ue[i] * (1 - U0[i, j + 1] / Ue[i])
                        deltatheta[i]   += (arg1 + arg2) / 2 * ((self.x[which_bloc][i, j + 1] - self.x[which_bloc][i, j]) ** 2 \
                                                            +   (self.y[which_bloc][i, j + 1] - self.y[which_bloc][i, j]) ** 2) ** 0.5
            self._deltatheta[offset:offset+self.nx[which_bloc]] = deltatheta #Assign delta_theta value on this block to overall array
            offset += self.nx[which_bloc]  # Update offset for next block

        return self._deltatheta
    

    def print(self):
        print("joy to the world")




    def extract_coordinates_and_velocity(self, U0=None):
        """
        Extracts the x-coordinate, y-coordinate, and x-component of the velocity for each grid point.

        Parameters:
        U0 (ndarray, optional): Velocity field array. If None, it is copied from stats.

        Returns:
        ndarray: A matrix where each entry contains the x-coordinate, y-coordinate,
                 and the x-component of the velocity.

        Raises:
        ValueError: If U0 is None and cannot be read from stats.
        """

        if U0 is None and self.is_stats:
            U0 = np.array(self._stats1[self.bloc[0]]["u"])

        if U0 is None:
            raise ValueError("U0 must be provided or read from stats.")

        result_matrix = []

        for which_bloc in self.bloc:
            for i in range(self.nx[which_bloc]):
                print("i range is", self.nx[which_bloc])
                for j in range(self.ny[which_bloc]):
                    print("size of U0 is",U0.shape)
                    print("j range is:",self.ny[which_bloc])
                    x = self.x[which_bloc][i, j]
                    y = self.y[which_bloc][i, j]
                    u = U0[i, j]
                    result_matrix.append([x, y, u])

        return np.array(result_matrix)


    # Calculatin of the boundary layer thickness associated Reynolds numbers:
    def calc_reynolds_bl(self):
        """
        Calculates the Reynolds numbers for boundary layer parameters.
    
        Returns:
            tuple: Reynolds numbers for delta_99, displacement thickness, and momentum thickness.
        """
        # Reference values
        rho_ref = self.dict_info["Roref"]
        Uref    = [self.dict_info["Mref"]*self.dict_info["cref"] for i in range(self._nx_tot)]
        mu_ref  = self.dict_info["Muref"]
        # Compute BL thickness: 

        # Re evolution
        Re_d99   = self._d99*Uref*rho_ref/mu_ref
        Re_dstar = self._deltas*Uref*rho_ref/mu_ref
        Re_theta = self._deltatheta*Uref*rho_ref/mu_ref
        return Re_d99, Re_dstar, Re_theta

    # ===============================================
    # Calculation of friction vel. utau and coeff. cf
    # ===============================================
    def flog(self, u, nuwall, Um, yc):
        """
        Logarithmic profile function for wall values calculation.

        Args:
            u (float): Velocity magnitude.
            nuwall (float): Wall kinematic viscosity.
            Um (float): Velocity at the first off-wall node.
            yc (float): Height from the wall.

        Returns:
            float: Calculated value.
        """
        z = 2.4 * np.log(yc * u / nuwall) + 5.24 - Um / u 
        return z

    def calc_utau_cf(self, data=None, duTdn=None, ilog=0, ut_ini=0, Ue=None):
        """
        Calculates friction velocity (utau), skin friction coefficient (Cf), and other wall values.

        Args:
            data (dict): Dictionary containing flow data.
            duTdn (array-like): Array containing the normal derivative of velocity at the wall.
            ilog (int): Flag for wall resolution method. If 0, resolved; if 1, unresolved.
            ut_ini (float): Initial value for utau calculation.

        Returns:
            tuple: Calculated values of tau_w, utau, Cf, Cf2, Re_tau, y_plus_x, y_plus_normal, x_plus_x, z_plus_x.
        """
        # Define references values:
        if Ue==None:
            U_ref   = self.dict_info["Mref"] * self.dict_info["cref"]
            Ue      = np.array([U_ref for _ in range(self._nx_tot)])

        rho_ref = self.dict_info["Roref"]

        offset = 0 # Offset to keep track of current position in arrays

        # Define wall arrays

        
        rho_w   = np.ndarray(self._nx_tot)
        mu_w    = np.ndarray(self._nx_tot)
        rho_e   = np.ndarray(self._nx_tot)


        # Define arrays for tau_w, utau, cf, re_tau:
        
        tau_wf, utauf, Cff, Re_tauf =   np.zeros(self._nx_tot), np.zeros(self._nx_tot), \
                                        np.zeros(self._nx_tot), np.zeros(self._nx_tot)

        
        

        # Calculate wall values:                    
        if self.dict_info["is_curv"] == "F":
            if self.is_stats:
                U0 = data["u"]

            a04 = np.zeros(5)
            a04[0] = -25. / 12.
            a04[1] = 4.
            a04[2] = -3.
            a04[3] = 4. / 3.
            a04[4] = -1. / 4.

            epsilon = 1.e-6
            tab = [50, 60, 70, 80]

            for which_bloc in self.bloc:
                rho_w   = np.zeros(self.nx[which_bloc])
                mu_w    = np.zeros(self.nx[which_bloc])
                rho_e   = np.zeros(self.nx[which_bloc])
                tau_w, utau, Cf, Re_tau =   np.zeros(self.nx[which_bloc]), np.zeros(self.nx[which_bloc]), \
                                            np.zeros(self.nx[which_bloc]), np.zeros(self.nx[which_bloc])
                for i in range(self.nx[which_bloc]):
                    if ilog == 1:
                        uuu = np.zeros(len(tab))
                        ut      = ut_ini
                        nuwall  = self._stats1[which_bloc]["mu"][i, 0] / self._stats1[which_bloc]["rho"][i, 0]
                        nu      = self._stats1[which_bloc]["mu"][i, 0] / self._stats1[which_bloc]["rho"][i, 0]

                        for k in range(len(tab)):
                            yc = tab[k] * nu / ut
                            j = 0
                            while self.y[self.bloc][j] < yc:
                                j += 1
                            Um = U0[i, j]
                            yc = self.y[self.bloc][j]
                            a = 0.001
                            b = Ue[offset+i] / 10.
                            if b == 0:
                                b = 2 * a
                            it = 1
                            erreur = 1
                            while erreur > epsilon:
                                c = (a + b) / 2.
                                if (self.flog(a, nuwall, Um, yc) * self.flog(c, nuwall, Um, yc) > 0):
                                    a = c
                                    b = b
                                else:
                                    a = a
                                    b = c

                                erreur = abs(self.flog(c, nuwall, Um, yc))
                                it = it + 1

                            uuu[k] = c

                        utau[i] = np.sum(uuu) / len(tab)
                        tau_w[i] = utau[i] ** 2 * self._stats1[which_bloc]["rho"][i, 0][i]
                        Cf[i] = tau_w[i] / (0.5 * self._stats1[which_bloc]["rho"][i, self.ny[which_bloc]-30][i] * Ue[offset+i] ** 2)
                        Re_tau[i] = self.d99[i] * utau[i] * self._stats1[which_bloc]["rho"][i, 0][i] / self._stats1[which_bloc]["mu"][i, 0]

                    else:
                        j = 0
                        dy4 = 0.
                        for l in range(4):
                            dy4 = dy4 + a04[1 + l] * self.y[which_bloc][j + l]
                        dudy = 0
                        for l in range(4):
                            dudy = dudy + a04[1 + l] * U0[i, j + l]

                        dudy = dudy / dy4

                        tau_w[i] = self._stats1[which_bloc]["mu"][i, 0] * dudy
                        if tau_w[i] < 0:
                            utau[i] = -np.sqrt(-tau_w[i] / self._stats1[which_bloc]["rho"][i, 0][i])
                        else:
                            utau[i] = np.sqrt(tau_w[i] / self._stats1[which_bloc]["rho"][i, 0][i])
                        Cf[i] = tau_w[i] / (0.5 * self._stats1[which_bloc]["rho"][i, self.ny[which_bloc]-30][i] * Ue[offset+i] ** 2)
                        Re_tau[i] = self._d99[i] * utau[i] * self._stats1[which_bloc]["rho"][i, 0][i] / self._stats1[which_bloc]["mu"][i, 0]
                        
                # Append values of variables in block to overall array
                tau_wf[offset:offset+self.nx[which_bloc]]   = tau_w
                utauf[offset:offset+self.nx[which_bloc]]    = utau
                Cff[offset:offset+self.nx[which_bloc]]      = Cf
                Re_tauf[offset:offset+self.nx[which_bloc]]  = Re_tau
                offset += self.nx[which_bloc]  # Update offset for next block

        else: # 2D or 3D Curvilinear solver
            # Define arrays:
            tau_w, utau, Cf, Re_tau = np.zeros(self._nx_tot), np.zeros(self._nx_tot), \
                                       np.zeros(self._nx_tot), np.zeros(self._nx_tot)
            
            
            # Calculate wall values
            offset = 0
            for which_bloc in self.bloc:
                if self.is_stats:
                    for i in range(self.nx[which_bloc]):
                        rho_w[offset +i]    = self._stats1[which_bloc]["rho"][i, 0]
                        mu_w [offset +i]    = self._stats1[which_bloc]["mu"][i, 0]
                        rho_e[offset +i]    = self._stats1[which_bloc]["rho"][i, self.ny[which_bloc]-30]
                        #print("size of nx is", self.nx[which_bloc])
                        #print("size of mu_w is", mu_w.size )
                    offset += self.nx[which_bloc]
            # Calculate wall values:
            #print("size of mu_w is", mu_w.size)
            #print("size of duTdn is", duTdn.size)
            tau_w   = mu_w*duTdn
            Cf      = tau_w/(0.5*rho_e*Ue**2)
            #print(Cf)
            for m in range(self._nx_tot):
                #if(tau_w[i]) < 0:
                if(tau_w[m]) < 0:
                    utau[m] = -np.sqrt(-tau_w[m] / rho_w[m])
                else:
                    utau[m] = np.sqrt(tau_w[m] / rho_w[m])
            Re_tau  = self._d99*utau*rho_w/mu_w
            #print(tau_w[m] / rho_w[m])
            #print(tau_w)
            #print(mu_w)
            #print(tau_w)
            #print(utau)

        return tau_w, utau, Cf, Re_tau
        
    def calc_turb_intensity(self, jmin_pos, jmax_pos, U_ref=None):
        """Calculate the turbulent intensity in the freestream.

        Args:
            jmin_pos (int): Minimum vertical index for turbulent intensity calculation.
            jmax_pos (int): Maximum vertical index for turbulent intensity calculation.

        Returns:
            tuple: Tuple containing turbulent intensity values.
                - Tu_sim: Average turbulent intensity.
                - Tu_x: Turbulent intensity along x direction
                - Tu_y: Turbulent intensity along y direction
                - Tu_z: Turbulent intensity along z direction
        """

        if not self._stats1:
            raise ValueError("Turbulence statistics data not available. Call read_stats_stbl method first.")

        if U_ref==None: U_ref = self.dict_info["Mref"] * self.dict_info["cref"]

        Tu_sim, Tu_x, Tu_y, Tu_z = [], [], [], []

        for which_bloc in self.bloc_fst:
            #if which_bloc not in self._stats1:
            #    continue

            jm_min, jm_max = self.ny[which_bloc] - jmin_pos, self.ny[which_bloc] - jmax_pos

            for i in range(self.nx[which_bloc]):
                u_mean = np.mean(self._stats1[which_bloc]['u'][i][jm_min:jm_max])
                v_mean = np.mean(self._stats1[which_bloc]['v'][i][jm_min:jm_max])
                w_mean = np.mean(self._stats1[which_bloc]['w'][i][jm_min:jm_max])

                u2_mean = np.mean(self._stats1[which_bloc]['u2'][i][jm_min:jm_max])
                v2_mean = np.mean(self._stats1[which_bloc]['v2'][i][jm_min:jm_max])
                w2_mean = np.mean(self._stats1[which_bloc]['w2'][i][jm_min:jm_max])

                Tu_sim.append(np.sqrt((u2_mean + v2_mean + w2_mean - u_mean**2 - v_mean**2 - w_mean**2) / 3) / U_ref * 100)
                Tu_x.append(np.sqrt(u2_mean - u_mean**2) / U_ref * 100)
                Tu_y.append(np.sqrt(v2_mean - v_mean**2) / U_ref * 100)
                Tu_z.append(np.sqrt(w2_mean - w_mean**2) / U_ref * 100)

        return Tu_sim, Tu_x, Tu_y, Tu_z
    

    # ==================
    #  Wall units, yplus, uplus
    # ==================
    
    def wall_length_scale(self):
        """ Calculate the wall unit length to scale the wall distance and velocity
        Args:
            None

        Returns:
            Numpy array: contains wall unit length for each point along x-axis
        """
        self.l_nu = np.zeros((self._nx_tot, self.ny[1]))
        offset = 0
        for which_block in self.bloc:
            # Calculate the friction velocity u_tau
            u_tau = np.sqrt(np.abs(self._stats2[which_block]["rho*duy"][:, :] * self._stats1[which_block]["mu"][:, :] / self._stats1[which_block]["rho"][:, :]**2))
            #u_tau = np.sqrt(np.abs(self._stats2[which_block]["rho*duy"][:, :] / self._stats1[which_block]["rho"][:, :]))
            # Calculate the wall unit length scale l_nu
            self.l_nu[offset:offset+self.nx[which_block], :] = self._stats1[which_block]["mu"][:, :] / (self._stats1[which_block]["rho"][:, :] * u_tau)
            offset += self.nx[which_block]
        return self.l_nu

    def wall_units(self):
        """ Calculates wall values, yplus and uplus
        Args:
            None

        Returns:
            Tuple: contains normalized wall distance y+ and normalized velocity u+
        """
        self.wall_length_scale()
        y_plus = np.zeros((self._nx_tot, self.ny[1]))
        u_plus = np.zeros((self._nx_tot, self.ny[1]))
        offset = 0
        for which_block in self.bloc:
            for j in range(self.ny[which_block]):
                # Calculate y_plus
                y_plus[offset:offset+self.nx[which_block], j] = self.y[which_block][:, j] / self.l_nu[offset:offset+self.nx[which_block], 0]
                # Calculate u_plus
                u_plus[offset:offset+self.nx[which_block], j] = self._stats1[which_block]["u"][:, j] / np.sqrt(self._stats2[which_block]["rho*duy"][:, 0] * self._stats1[which_block]["mu"][:, 0]/ self._stats1[which_block]["rho"][:, 0]**2)
                #u_plus[offset:offset+self.nx[which_block], j] = self._stats1[which_block]["u"][:, j] / np.sqrt(self._stats2[which_block]["rho*duy"][:, 0]/ self._stats1[which_block]["rho"][:, 0])
            offset += self.nx[which_block]
        return y_plus, u_plus

