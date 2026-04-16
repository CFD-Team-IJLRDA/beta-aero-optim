import sys
import numpy as np


class readResultsMusicaa:
    """ Class to read data from simulation in MUSICAA, this class aims at generalizing the post_Pocessign procedure in musicaa
        rep: repository to postProcess 
        ngh: number of ghost points 
        is_extended: is the grid file with ghost points or without
        Gamma: ratio of specific heats, initialize for air
        r_gas: gas constant, default is r_air
        is_2D: is the flow 2 or 3 dimensional, default is 2D
        is_RANS: is it a RANS simulation or not, default is false
        model_RANS: RANS model if is RANS simulation"""
    def __init__(self,
                 rep,
                 ngh,
                 is_extended,
                 gamma=1.4,
                 r_gas=287.06,
                 is_little_endian=True,
                 is_2D=False,      
                 is_RANS=False,
                 model_RANS=None) -> None:
        self.repo               = rep
        self.ngh                = ngh
        self.is_extended        = is_extended
        self.gamma              = gamma 
        self.r_gas              = r_gas
        self.is_little_endian   = is_little_endian
        self.is_2D              = is_2D
        self.is_RANS            = is_RANS
        self.model_RANS         = model_RANS
        if self.is_RANS == True and self.model_RANS == None: 
            raise SystemError("RANS model must be specified when postProcessing a RANS simulation")
        self.dict_info      = {} 
        self.dict_time      = { "timestamp" : [],
                                "ite": [],
                                "time": []}
        self.dict_infoBl    = {}
        self.dict_infoVol={} ; self.dict_infoPlane={}
        self.dict_infoLine={}; self.dict_infoPoint={}
        if is_2D and is_RANS and model_RANS=="SA" or model_RANS=="SA_gamma_re_theta":
            self.residual = {'nn': [], 'Rho': [], 'Rhou': [], 'Rhov': [], 'Rhoe': [], "nutil": []}
        elif is_2D and not is_RANS:
            self.resdual = {'nn': [], 'Rho': [], 'Rhou': [], 'Rhov': [], 'Rhoe': []}
        elif not is_2D and not is_RANS:
            self.residual = {'nn': [], 'Rho': [], 'Rhou': [], 'Rhov': [], 'Rhow': [], 'Rhoe': []}
        self.nx, self.ny, self.nz, self.x, self.y, self.z               = {},{},{},{},{},{}
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax= {},{},{},{},{},{}
        self._stats1     = {}
        self._stats2     = {}
        self.planes = {}


    
    # =============
    # Read info.ini
    # =============
    def read_info(self,
                  is_chan=False):
        """Reads the case data in info.ini given the repository name in from repository
        if the case is a 2D channel flow, set is_chan to True when calling the function"""
        l = 0
        file_input  = self.repo+"/info.ini"
        try:
            f = open(file_input,'r')
            print("*======")
            print("Reading info.ini")
        except FileNotFoundError:
            print("*======")
            exit("File info.ini not found")
        lines = f.readlines()
        self.dict_info["nbloc"]  = int(lines[0].split()[4])
        self.dict_info["is_curv"]= lines[0].split()[5]
        # Compatibility old and new info files
        if lines[1].split()[3]!='=': l=1
        for ind in range(self.dict_info["nbloc"]):
            self.dict_info["nx_bl"+str(ind+1)] = int(lines[1+ind].split()[4+l])
            self.dict_info["ny_bl"+str(ind+1)] = int(lines[1+ind].split()[5+l])
            self.dict_info["nz_bl"+str(ind+1)] = int(lines[1+ind].split()[6+l])
        self.dict_info["etot0"]  = float(lines[2+ind].split()[3])
        self.dict_info["mgtot0"] = float(lines[2+ind].split()[4])
        self.dict_info["xmin"]   = float(lines[3+ind].split()[4])
        self.dict_info["ymin"]   = float(lines[3+ind].split()[5])
        self.dict_info["zmin"]   = float(lines[3+ind].split()[6])
        self.dict_info["xmax"]   = float(lines[4+ind].split()[4])
        self.dict_info["ymax"]   = float(lines[4+ind].split()[5])
        self.dict_info["zmax"]   = float(lines[4+ind].split()[6])
        self.dict_info["Mref"]   = float(lines[5+ind].split()[3])
        self.dict_info["Reref"]  = float(lines[5+ind].split()[4])
        self.dict_info["Mupref"] = float(lines[6+ind].split()[3])
        self.dict_info["Muref"]  = float(lines[6+ind].split()[4])
        self.dict_info["Roref"]  = float(lines[7+ind].split()[4])
        self.dict_info["Pref"]   = float(lines[7+ind].split()[5])
        self.dict_info["Tref"]   = float(lines[7+ind].split()[6])
        self.dict_info["Uref"]   = float(lines[8+ind].split()[4])
        self.dict_info["cref"]   = float(lines[8+ind].split()[5])
        self.dict_info["Tscale"] = float(lines[8+ind].split()[6])
        self.dict_info["Uref"]   = self.dict_info["Mref"]*self.dict_info["cref"]
        if len(lines)==10+ind: self.dict_info["deltat"] = float(lines[9+ind].split()[4])
        if is_chan: self.dict_info["forcing"] = float(lines[10+ind].split()[3])
        print("Done reading info.ini")


    # =============
    # Read time.ini
    # =============
    def read_time(self):
        try:
            f = open(self.repo+"/time.ini",'r')
            print("*======")
            print("Reading time.ini")
        except FileNotFoundError:
            print("File time.ini not found")
        lines = f.readlines()[1:]
        for line in lines:
            line = line.split()
            self.dict_time["timestamp"].append(line[0])
            self.dict_time["ite"].append(int(line[2]))
            self.dict_time["time"].append(float(line[4]))
        print("Done reading time.ini")

    # ====================
    # Read param_block.ini
    # ====================
    def read_param_blocks(self,
                          verbose=True):

        try:
            f=open(self.repo+"/param_blocks.ini",'r')
            print("*======")
            print("Reading param_blocks.ini")
            self.dict_infoBl,ind = {},0
        except FileNotFoundError:
            print("*======")
            print("File param_blocks.ini not found")
            self.dict_infoBl,ind = None,-1

        if ind==0:
            lines = f.readlines()
            line = lines[0]
            while line[0]=='!': ind += 1; line = lines[ind]
            self.dict_infoBl['nblc'] = int(line.split()[0])
            nblc = self.dict_infoBl['nblc']
            ind += 1; line = lines[ind]
            for ib in range(1,nblc+1):
                self.dict_infoBl[ib] = {}
                # ----------
                while line[0]=='!': ind += 1; line = lines[ind]
                # Nb of points and nb of procs
                # ----------------------------
                self.dict_infoBl[ib]['i_pt'] = int(line.split()[0])
                self.dict_infoBl[ib]['i_proc'] = int(line.split()[1])
                ind += 1; line = lines[ind]
                self.dict_infoBl[ib]['j_pt'] = int(line.split()[0])
                self.dict_infoBl[ib]['j_proc'] = int(line.split()[1])
                ind += 1; line = lines[ind]
                self.dict_infoBl[ib]['k_pt'] = int(line.split()[0])
                self.dict_infoBl[ib]['k_proc'] = int(line.split()[1])
                ind += 1; line = lines[ind]
                # ----------
                while line[0]=='!': ind += 1; line = lines[ind]
                # Boundary conditions
                # -------------------
                ind += 1; line = lines[ind]
                self.dict_infoBl[ib]['bc_imin'] = int(line.split()[0])
                self.dict_infoBl[ib]['bc_imax'] = int(line.split()[1])
                self.dict_infoBl[ib]['bc_jmin'] = int(line.split()[2])
                self.dict_infoBl[ib]['bc_jmax'] = int(line.split()[3])
                self.dict_infoBl[ib]['bc_kmin'] = int(line.split()[4])
                self.dict_infoBl[ib]['bc_kmax'] = int(line.split()[5])
                ind += 1; line = lines[ind]
                # BC type
                self.dict_infoBl[ib]['bc_imin_t'] = str(line.split()[0])
                self.dict_infoBl[ib]['bc_imax_t'] = str(line.split()[1])
                self.dict_infoBl[ib]['bc_jmin_t'] = str(line.split()[2])
                self.dict_infoBl[ib]['bc_jmax_t'] = str(line.split()[3])
                self.dict_infoBl[ib]['bc_kmin_t'] = str(line.split()[4])
                self.dict_infoBl[ib]['bc_kmax_t'] = str(line.split()[5])
                ind += 1; line = lines[ind]
                #----------- New version of param_block.ini
                if line[0]!='!': ind += 1; line = lines[ind]
                # ----------
                while line[0]=='!': ind += 1; line = lines[ind]
                # Sponge zone
                # -----------
                ind += 2; line = lines[ind]
                # ----------
                while line[0]=='!': ind += 1; line = lines[ind]
                # Output snapshots
                # ----------------
                self.dict_infoBl[ib]['nb_sn'] = int(line.split()[0])
                ind += 1; line = lines[ind]
                while line[0]=='!': ind += 1; line = lines[ind]
                self.dict_infoBl[ib]['sn_numbers'] = []
                for num_sn in range(1,self.dict_infoBl[ib]['nb_sn']+1):
                    self.dict_infoBl[ib]['sn_numbers'].append(num_sn)
                    self.dict_infoBl[ib][num_sn] = {}
                    self.dict_infoBl[ib][num_sn]['I1'] = int(line.split()[0]); self.dict_infoBl[ib][num_sn]['I2'] = int(line.split()[1])
                    self.dict_infoBl[ib][num_sn]['J1'] = int(line.split()[2]); self.dict_infoBl[ib][num_sn]['J2'] = int(line.split()[3])
                    self.dict_infoBl[ib][num_sn]['K1'] = int(line.split()[4]); self.dict_infoBl[ib][num_sn]['K2'] = int(line.split()[5])
                    self.dict_infoBl[ib][num_sn]['freq'] = int(line.split()[6])
                    self.dict_infoBl[ib][num_sn]['nvar'] = int(line.split()[7])
                    n_udf = 1
                    for ivar in range(1,self.dict_infoBl[ib][num_sn]['nvar']+1):
                        self.dict_infoBl[ib][num_sn]['var'+str(ivar)] = str(line.split()[7+ivar])
                        if str(line.split()[7+ivar])=='udf':
                            self.dict_infoBl[ib][num_sn]['var'+str(ivar)] = str(line.split()[7+ivar])+str(n_udf)
                            n_udf += 1
                    ind += 1; line = lines[ind]
                while line[0]!='!': ind += 1; line = lines[ind]
                while line[0]=='!' and ind<len(lines)-1: ind += 1; line = lines[ind]
            print("Done reading param_blocks.ini")
    
    # ==========================
    # Read volumes/planes/lines:
    # ==========================
    def snap_type_characteristics(self):
        # Determination of the snapshot types and type characteristics
        # to get informations by type of data [point, line, plane or volume]
        print("*======")
        print("Read snapshot characteristics")
        nblc = self.dict_infoBl['nblc']
        

        for ib in range(1,nblc+1):
            # Initialization of number of volumes, planes, lines and points
            self.dict_infoBl[ib]['nb_v']=0; self.dict_infoBl[ib]['nb_p']=0
            self.dict_infoBl[ib]['nb_l']=0; self.dict_infoBl[ib]['nb_pt']=0
            # self.dict_infoBl[ib]['p_numbers'] = []
            self.dict_infoVol[ib]={}; self.dict_infoPlane[ib]={}
            self.dict_infoLine[ib]={}; self.dict_infoPoint[ib]={}

            if self.dict_infoBl[ib]['nb_sn']==0: continue

            # Loop on the number of snapshot
            for num_sn in range(1,self.dict_infoBl[ib]['nb_sn']+1):
                # Test on the indices
                i1=self.dict_infoBl[ib][num_sn]['I1']; i2=self.dict_infoBl[ib][num_sn]['I2']
                j1=self.dict_infoBl[ib][num_sn]['J1']; j2=self.dict_infoBl[ib][num_sn]['J2']
                k1=self.dict_infoBl[ib][num_sn]['K1']; k2=self.dict_infoBl[ib][num_sn]['K2']
                # Point
                if i1==i2 and j1==j2 and k1==k2:
                    self.dict_infoBl[ib]['nb_pt'] += 1
                    self.dict_infoBl[ib][num_sn]['type'] = 0
                    ipt = self.dict_infoBl[ib]['nb_pt']
                    self.dict_infoPoint[ib][ipt] = {}
                    self.dict_infoPoint[ib][ipt]['I1']=i1; self.dict_infoPoint[ib][ipt]['I2']=i2
                    self.dict_infoPoint[ib][ipt]['J1']=j1; self.dict_infoPoint[ib][ipt]['J2']=j2
                    self.dict_infoPoint[ib][ipt]['K1']=k1; self.dict_infoPoint[ib][ipt]['K2']=k2
                    self.dict_infoPoint[ib][ipt]['nvar']=self.dict_infoBl[ib][num_sn]['nvar']
                    for ivar in range(1,self.dict_infoBl[ib][num_sn]['nvar']+1):
                        self.dict_infoPoint[ib][ipt]['var'+str(ivar)] = self.dict_infoBl[ib][num_sn]['var'+str(ivar)]
                # Line
                elif (i1==i2 and j1==j2) or (k1==k2 and j1==j2) or (i1==i2 and k1==k2):
                    self.dict_infoBl[ib]['nb_l'] += 1
                    self.dict_infoBl[ib][num_sn]['type'] = 1
                    il = self.dict_infoBl[ib]['nb_l']
                    self.dict_infoLine[ib][il] = {}
                    self.dict_infoLine[ib][il]['I1']=i1; self.dict_infoLine[ib][il]['I2']=i2
                    self.dict_infoLine[ib][il]['J1']=j1; self.dict_infoLine[ib][il]['J2']=j2
                    self.dict_infoLine[ib][il]['K1']=k1; self.dict_infoLine[ib][il]['K2']=k2
                    self.dict_infoLine[ib][il]['nvar']=self.dict_infoBl[ib][num_sn]['nvar']
                    for ivar in range(1,self.dict_infoBl[ib][num_sn]['nvar']+1):
                        self.dict_infoLine[ib][il]['var'+str(ivar)] = self.dict_infoBl[ib][num_sn]['var'+str(ivar)]
                    if i1==i2 and j1==j2:
                        self.dict_infoLine[ib][il]['dir']=3
                    elif i1==i2 and k1==k2:
                        self.dict_infoLine[ib][il]['dir']=2
                    elif j1==j2 and k1==k2:
                        self.dict_infoLine[ib][il]['dir']=1
                # Plane
                elif i1==i2 or j1==j2 or k1==k2:
                    self.dict_infoBl[ib]['nb_p'] += 1
                    # self.dict_infoBl[ib]['p_numbers'].append(self.dict_infoBl[ib]['nb_p']) # Useless, kept for compatibility
                    self.dict_infoBl[ib][num_sn]['type'] = 2
                    ip = self.dict_infoBl[ib]['nb_p']
                    self.dict_infoPlane[ib][ip] = {}
                    self.dict_infoPlane[ib][ip]['I1']=i1; self.dict_infoPlane[ib][ip]['I2']=i2
                    self.dict_infoPlane[ib][ip]['J1']=j1; self.dict_infoPlane[ib][ip]['J2']=j2
                    self.dict_infoPlane[ib][ip]['K1']=k1; self.dict_infoPlane[ib][ip]['K2']=k2
                    self.dict_infoPlane[ib][ip]['nvar']=self.dict_infoBl[ib][num_sn]['nvar']
                    for ivar in range(1,self.dict_infoBl[ib][num_sn]['nvar']+1):
                        self.dict_infoPlane[ib][ip]['var'+str(ivar)] = self.dict_infoBl[ib][num_sn]['var'+str(ivar)]
                    if i1==i2:
                        self.dict_infoPlane[ib][ip]['normal']=1
                        self.dict_infoPlane[ib][ip]['index']=i1
                    elif j1==j2:
                        self.dict_infoPlane[ib][ip]['normal']=2
                        self.dict_infoPlane[ib][ip]['index']=j1
                    elif k1==k2:
                        self.dict_infoPlane[ib][ip]['normal']=3
                        self.dict_infoPlane[ib][ip]['index']=k1
                else:
                    self.dict_infoBl[ib]['nb_v'] += 1
                    self.dict_infoBl[ib][num_sn]['type'] = 3
                    iv = self.dict_infoBl[ib]['nb_v']
                    self.dict_infoVol[ib][iv] = {}
                    self.dict_infoVol[ib][iv]['I1']=i1; self.dict_infoVol[ib][iv]['I2']=i2
                    self.dict_infoVol[ib][iv]['J1']=j1; self.dict_infoVol[ib][iv]['J2']=j2
                    self.dict_infoVol[ib][iv]['K1']=k1; self.dict_infoVol[ib][iv]['K2']=k2
                    self.dict_infoVol[ib][iv]['nvar']=self.dict_infoBl[ib][num_sn]['nvar']
                    for ivar in range(1,self.dict_infoBl[ib][num_sn]['nvar']+1):
                        self.dict_infoVol[ib][iv]['var'+str(ivar)] = self.dict_infoBl[ib][num_sn]['var'+str(ivar)]
        print("Done reading snapshot characteristics")

    # ===============
    # Read residuals:
    # ===============
    def read_residuals( self,
                        filename):
        # Reading of the residuals
        # ------------------------
        read = True
        try:
            f = open(self.repo+"/"+filename,'rb')
            print("*======")
            print("Reading residuals.")
        except FileNotFoundError:
            print("*======")
            print(f"File {filename} not found, check filename of residual file")
            read=False
        while read:
            try:

                #marker1 = np.fromfile(f, dtype='>i4',count=1)[0]

                #arg = np.fromfile(f, dtype=('>i4'), count=1)
                self.residual['nn'].append(np.fromfile(f, dtype=('>i4'), count=1)[0])
                #==============================================
                #arg = np.fromfile(f, dtype=('>f8'), count=1)
                self.residual['Rho'].append(np.fromfile(f, dtype=('>f8'), count=1)[0])
                #==============================================
                #arg = np.fromfile(f, dtype=('>f8'), count=1)
                self.residual['Rhou'].append(np.fromfile(f, dtype=('>f8'), count=1)[0])
                #==============================================
                #arg = np.fromfile(f, dtype=('>f8'), count=1)
                self.residual['Rhov'].append(np.fromfile(f, dtype=('>f8'), count=1)[0])
                #==============================================
                self.residual['Rhoe'].append(np.fromfile(f, dtype=('>f8'), count=1)[0])
                #==============================================
                if self.is_2D and self.is_RANS :
                    #arg = np.fromfile(f, dtype=('>f8'), count=1)
                    self.residual['nutil'].append(np.fromfile(f, dtype=('>f8'), count=1)[0])
                elif not self.is_2D and not self.is_RANS :
                    #arg = np.fromfile(f, dtype=('>f8'), count=1)
                    self.residual['Rhow'].append(np.fromfile(f, dtype=('>f8'), count=1)[0])
                    #arg = np.fromfile(f, dtype=('>f8'), count=1)
                    self.residual['Rhoe'].append(np.fromfile(f, dtype=('>f8'), count=1)[0])

                #marker2 = np.fromfile(f,dtype='>i4',count=1)[0]
 
            except IndexError:
                break
        print("Done reading residuals")
        return self.residual

    # =========
    # Read grid 
    # =========
    def read_grid_block(self, 
                        filename, 
                        verbose=False,
                        is_extended=False,
                        is_sw_rv=False):
        """Reads the grid from one block given the filename and other parameters.

        Args:
            filename (str): Name of the binary file containing the grid data.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            is_extended (bool, optional): True if the flow is 2D or statistically 2D. Defaults to False.
            is_sw_rv (bool, optional): TBD. Defaults to False.

        Returns:
            tuple: Tuple containing grid data.
        """
        # Reading of the grid
        # -------------------

        sens = '>'
        if self.is_little_endian: sens = '<'

        try: 
            f = open(self.repo+"/"+filename, "r")
        except FileNotFoundError:
            raise SystemExit("File %s not found..."%(filename))
        
        arg = np.fromfile(f, dtype=(sens+'i4'), count=1)
        nx  = np.fromfile(f, dtype=(sens+'i4'), count=1)[0]
        #===============================================================================
        arg = np.fromfile(f, dtype=(sens+'i8'), count=1)
        ny  = np.fromfile(f, dtype=(sens+'i4'), count=1)[0]
        #===============================================================================
        arg = np.fromfile(f, dtype=(sens+'i8'), count=1)
        if not is_extended:
            nz  = np.fromfile(f, dtype=(sens+'i4'), count=1)[0]
            #===============================================================================
            arg = np.fromfile(f, dtype=(sens+'i8'), count=1)
        if self.dict_info["is_curv"] == "T":
            x = np.zeros((nx, ny),dtype='float',order='F')
            for j in range(ny):
                x[:,j] = np.fromfile(f, dtype=(sens+'f8'), count=nx)
            #===============================================================================
            arg = np.fromfile(f, dtype=(sens+'i8'), count=1)
            y = np.zeros((nx, ny),dtype='float',order='F')
            for j in range(ny):
                y[:,j] = np.fromfile(f, dtype=(sens+'f8'), count=nx)
            #===============================================================================
            if not is_extended:
                arg = np.fromfile(f, dtype=(sens+'i8'), count=1)
                z = np.fromfile(f, dtype=(sens+'f8'), count=nz)
                #===============================================================================
            if is_sw_rv:
                swp_rv = np.zeros((4,3),dtype=int)
                # imin, imax, jmin, jmax
                for i in range(4):
                    # swap, reverse dir i, reverse dir j
                    for j in range(3):
                        arg = np.fromfile(f, dtype=(sens+'i8'), count=1)
                        swp_rv[i,j] = np.fromfile(f, dtype=(sens+'i4'), count=1)[0]
        else:
            x = np.fromfile(f, dtype=(sens+'f8'), count=nx)
            #===============================================================================
            arg = np.fromfile(f, dtype=(sens+'i8'), count=1)
            y = np.fromfile(f, dtype=(sens+'f8'), count=ny)
            #===============================================================================
            if not is_extended:
                arg = np.fromfile(f, dtype=(sens+'i8'), count=1)
                z = np.fromfile(f, dtype=(sens+'f8'), count=nz)
                #===============================================================================
        f.close()
        if is_extended:
            return nx, ny, x, y 
        else:
            
            return nx, ny, nz, x, y, z
    
    def read_grid_block_extended(self, 
                        filename, 
                        ngi, 
                        ngj,
                        ngk,
                        verbose=False,
                        is_sw_rv=False):
        
        """Reads the extended grid from one block given the filename and other parameters.

        Args:
            filename (str): Name of the binary file containing the grid data.
            ngi, ngj, ngk = dimension of the block (number of points i,j,k)
            ngh: number of ghost point
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            is_sw_rv (bool, optional): TBD. Defaults to False.

        Returns:
            tuple: Tuple containing grid data.
        """

        sens = '>'
        if self.is_little_endian: sens = '<'

        # Reading of the grid
        # -------------------

        try: 
            f = open(self.repo+"/"+filename, "r")
        except FileNotFoundError:
            raise SystemExit("File %s not found..."%(filename))
        if self.dict_info["is_curv"] == "F": 
            x = np.fromfile(f, dtype=(sens+'f8'), count=ngi).reshape((ngi), order='F')
            y = np.fromfile(f, dtype=(sens+'f8'), count=ngj).reshape((ngj), order='F')
            z = np.fromfile(f, dtype=(sens+'f8'), count=ngk).reshape((ngk), order='F')
        elif self.dict_info["is_curv"]=="T" and self.is_2D:
            x = np.fromfile(f, dtype=(sens+'f8'), count=ngi*ngj).reshape((ngi,ngj), order='F')
            y = np.fromfile(f, dtype=(sens+'f8'), count=ngi*ngj).reshape((ngi,ngj), order='F')
            print("ngk is:", ngk)
            if ngk>1 and ngk != 0:
                z = np.fromfile(f, dtype=('f8'), count=ngk).reshape((ngk), order='F')
            elif ngk==0:
                z = np.fromfile(f, dtype=('f8'), count=ngk).reshape((1), order='F')
            else:
                z=np.ndarray(1); z[0]=0.0
        elif self.dict_info["is_curv"]=="T" and not self.is_2D:
            x = np.fromfile(f, dtype=(sens+'f8'), count=ngi*ngj*ngk).reshape((ngi,ngj,ngk), order='F')
            y = np.fromfile(f, dtype=(sens+'f8'), count=ngi*ngj*ngk).reshape((ngi,ngj,ngk), order='F')
            z = np.fromfile(f, dtype=(sens+'f8'), count=ngi*ngj*ngk).reshape((ngi,ngj,ngk), order='F')
        else:
            raise SystemExit("Wrong choice for isolver in read_grid_new !")
        f.close()            
        return x,y,z
    
    def read_grid(self, 
                  verbose=False,
                  is_sw_rv=False):
        """Reads the extended grid for all blocks of the domain and returns a grid without ghost points

        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            is_sw_rv (bool, optional): TBD. Defaults to False.

        Returns:
            dict: Dictionary containing grid data for each block.
        """

        xmin_,xmax_,ymin_,ymax_ = 1000,-1000,1000,-1000
        # Extended grids: 
        nx_ex,ny_ex,nz_ex = {},{},{}
        x_ex,y_ex,z_ex = {},{},{}
        print("*======")
        print("Reading grid")
        if self.is_extended:
            for i in range(1,self.dict_info["nbloc"]+1):
                grid = 'grid_bl'+str(i)+"_ngh"+str(self.ngh)+'.bin'
                # Get nx, ny, nz for each block form info.ini:
                self.nx[i]=self.dict_info["nx_bl%i"%(i)]; self.ny[i]=self.dict_info["ny_bl%i"%(i)]; self.nz[i]=self.dict_info["nz_bl%i"%(i)]
                # Define extended borns: init
                nx_ex[i]=self.nx[i]+2*self.ngh
                ny_ex[i]=self.ny[i]+2*self.ngh
                if self.nz[i] > 1:
                    nz_ex[i]=self.nz[i]+2*self.ngh
                elif self.nz[i] == 1:
                    nz_ex[i]=self.nz[i]
                else:
                    raise SystemExit("Case of nz for block %i is unkown"%(i))
                # Read extended x,y,z:
                x_ex[i],y_ex[i],z_ex[i] = self.read_grid_block_extended(grid,nx_ex[i],ny_ex[i],nz_ex[i],verbose=True)
                # Grid without ghost points
                if self.dict_info["is_curv"]=="F": 
                    self.x[i] = x_ex[i][self.ngh:-self.ngh]; self.y[i] = y_ex[i][self.ngh:-self.ngh]
                    if self.nz[i] >1: self.z[i] = z_ex[i][self.ngh:-self.ngh]
                elif self.dict_info["is_curv"]=="T" and self.is_2D:
                    self.x[i] = x_ex[i][self.ngh:-self.ngh,self.ngh:-self.ngh]; self.y[i] = y_ex[i][self.ngh:-self.ngh,self.ngh:-self.ngh]
                    if self.nz[i] >1: self.z[i] = z_ex[i][self.ngh:-self.ngh]
                else:
                    self.x[i] = x_ex[i][self.ngh:-self.ngh,self.ngh:-self.ngh, self.ngh:-self.ngh]
                    self.y[i] = y_ex[i][self.ngh:-self.ngh,self.ngh:-self.ngh, self.ngh:-self.ngh]
                    self.z[i] = z_ex[i][self.ngh:-self.ngh,self.ngh:-self.ngh, self.ngh:-self.ngh]
                    # Some general charectiristic of the grid
                print("Block #%s, Mesh size: nx = %s, ny = %s, nz = %s"%(i,self.nx[i],self.ny[i],self.nz[i]))
        else:
            for i in range(1,self.dict_info["nbloc"]+1):
                #print(" Block is:", i)
                grid = 'grid_bl'+str(i)+'.bin'
                # Check existence of file
                try: 
                    f = open(self.repo+"/"+grid)
                    f.close()
                except FileNotFoundError:
                    exit(f"Error ! Cannot find file {grid}")
                if self.dict_info["is_curv"]=='F':
                    if self.is_2D:
                        self.nx[i], self.ny[i],self.x[i],self.y[i] = \
                            self.read_grid_block(grid, verbose=True, is_extended=self.is_2D, 
                                                 is_sw_rv=is_sw_rv)
                        (self.xmin[i], self.xmax[i],self.ymin[i],self.ymax[i]) = \
                            (min(self.x[i]),max(self.x[i]),min(self.y[i]),max(self.y[i]))
                    else:
                        self.nx[i], self.ny[i], self.nz[i],self.x[i],self.y[i],self.z[i] = \
                            self.read_grid_block(grid, verbose=True, is_extended=self.is_2D, 
                                                 is_sw_rv=is_sw_rv)
                        (self.xmin[i], self.xmax[i],self.ymin[i],self.ymax[i], self.zmin[i], self.zmax[i]) = \
                            (min(self.x[i]),max(self.x[i]),min(self.y[i]),max(self.y[i]),min(self.z[i]),max(self.z[i]))
                else:
                    x_read, y_read, z_read  = {}, {}, {}
                    self.nx[i], self.ny[i], self.nz[i], x_read[i],y_read[i],z_read[i] = \
                        self.read_grid_block(grid, verbose=True, 
                                             is_sw_rv=is_sw_rv)

                    if self.dict_info["is_curv"]=="F":
                        if not self.is_2D: 
                            self.x[i] = x_read[i][:, 0]
                            self.y[i] = y_read[i][0, :]
                            self.z[i] = z_read[i][:]
                        else:
                            self.x[i] = x_read[i][:, 0, 0]
                            self.y[i] = y_read[i][0, :, 0]
                            self.z[i] = z_read[i][0, 0, :]
                    else:
                        self.x[i] = x_read[i]
                        self.y[i] = y_read[i]
                        self.z[i] = z_read[i]

                    (self.xmin[i], self.xmax[i], self.ymin[i], self.ymax[i], self.zmin[i], self.zmax[i]) = \
                        (np.min(self.x[i]),np.max(self.x[i]),np.min(self.y[i]),np.max(self.y[i]),
                         np.min(self.z[i]),np.max(self.z[i]))
           
    # ===============
    # Read planes:
    # ===============
    def read_plane_block(self, 
                         filename, 
                         n1, 
                         n2, 
                         nvar, 
                         nplanes=-1,
                         num_p=0,
                         verbose=True):
        """Reads a plane for one block from a binary file.

        Args:
            filename (str): Name of the binary file containing the plane data.
            n1 (int): Size of dimension 1 of the plane.
            n2 (int): Size of dimension 2 of the plane.
            nvar (int): Number of variables in the plane.
            nplanes (int, optional): Number of planes to read (-1 to read until end of file). Defaults to -1.
            num_p (int, optional): Offset for the number of planes to read. Defaults to 0.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.

        Returns:
            dict, dict: Dictionaries containing plane data and information about the plane.
        """

        # Determination of the plane name
        #if 0<ip and ip<10:
        #    filename = self.repo+'/plane_00'+str(ip)+'_bl'+str(ib)+'.bin'
        #else:
        #    filename = self.repo+'/plane_0'+str(ip)+'_bl'+str(ib)+'.bin'
        
        try:
            f=open(filename,'r')
            self.plane, self.ind = {},0
        except FileNotFoundError:
            if verbose: print("File %s not found..."%(filename))
            self.plane, self.ind = None,-1

        for i in range(1,nvar+1): self.plane['var'+str(i)] = {}

        offset = num_p*n1*n2*nvar*8
        ind = -1
        if nplanes==-1:
            while ind>=-1:
                try:
                    for i in range(1,nvar+1):
                        self.plane['var'+str(i)][self.ind] = np.fromfile(f, dtype=('<f8'), count=n1*n2).reshape((n1,n2), order='F')
                        # plane['var'+str(i)][ind] = np.fromfile(f, dtype=('<f8'), count=n1*n2).reshape((n2,n1), order='C')
                    self.ind += 1
                except ValueError:
                    break
         
        else:
            try:
                self.plane['var1'][ind] = np.fromfile(f, dtype=('<f8'), count=n1*n2, offset=offset).reshape((n1,n2), order='F')
                for i in range(2,nvar+1):
                    self.plane['var'+str(i)][self.ind] = np.fromfile(f, dtype=('<f8'), count=n1*n2).reshape((n1,n2), order='F')
                self.ind += 1
            except ValueError:
                print("Value error")
                
            while self.ind<=nplanes:
                try:
                    for i in range(1,nvar+1):
                        self.plane['var'+str(i)][self.ind] = np.fromfile(f, dtype=('<f8'), count=n1*n2).reshape((n1,n2), order='F')
                    ind += 1
                except ValueError:
                    break
        return self.plane, self.dict_infoPlane
    
    def read_plane(self):
        """ Reads planes for each block of the domain.

            Returns:
                dict: Dictionary containing plane data for each block.
        """
        # Loop on blocks
        print("*======")
        print("Reading planes")
        for ib in range(1,self.dict_infoBl['nblc']+1):
            if self.dict_infoBl[ib]['nb_p']!=0: self.planes[ib] = {}
            # Loop on planes per block
            for ip in range(1,self.dict_infoBl[ib]['nb_p']+1):
                #if ip>3: continue
                # Creation of a dictionnary for each plane of each block
                self.planes[ib][ip] = {}
                if self.dict_infoPlane[ib][ip]['normal']==1: n1=self.ny[ib];n2=self.nz[ib]
                if self.dict_infoPlane[ib][ip]['normal']==2: n1=self.nx[ib];n2=self.nz[ib]
                if self.dict_infoPlane[ib][ip]['normal']==3: n1=self.nx[ib];n2=self.ny[ib]
                nvar = self.dict_infoPlane[ib][ip]['nvar']
                self.planes[ib][ip]['nb_p'] = self.dict_infoBl[ib]['nb_p']
                # Determination of the plane name
                if 0<ip and ip<10:
                    filename = self.repo+'/plane_00'+str(ip)+'_bl'+str(ib)+'.bin'
                else:
                    filename = self.repo+'/plane_0'+str(ip)+'_bl'+str(ib)+'.bin'
                temp,self.planes[ib][ip]['nb_save'] = self.read_plane_block(filename=filename, n1=n1, n2=n2, nvar=nvar, verbose=True)
                for i in range(1,nvar+1):
                    varname = self.dict_infoPlane[ib][ip]['var'+str(i)]
                    self.planes[ib][ip][varname] = temp['var'+str(i)]
        print("Done reading planes")            
        return self.planes
    
    # ==================
    # Read restart files
    # ==================
    def read_restart_block(self, 
                           file_input,
                           nx,
                           ny,
                           nz):
        """Reads restart file for one block"""
        try:
            f=open(file_input,'r')
            print("Reading %s..."%(file_input))
            ind = 0
        except FileNotFoundError:
            print("File %s not found..."%(file_input))
            ind = -1
            ro,rou,rov,row,roe = None,None,None,None,None
    
        if ind==0:
            f=open(file_input,'r')
    
            ro  = np.fromfile(f, dtype=('<f8'), count=nx*ny*nz).reshape((nx,ny,nz), order='F')
            rou = np.fromfile(f, dtype=('<f8'), count=nx*ny*nz).reshape((nx,ny,nz), order='F')
            rov = np.fromfile(f, dtype=('<f8'), count=nx*ny*nz).reshape((nx,ny,nz), order='F')
            row = np.fromfile(f, dtype=('<f8'), count=nx*ny*nz).reshape((nx,ny,nz), order='F')
            roe = np.fromfile(f, dtype=('<f8'), count=nx*ny*nz).reshape((nx,ny,nz), order='F')
    
            if self.is_2D:
                ro_,rou_,rov_,row_,roe_ = ro,rou,rov,row,roe
                ro,rou,rov,row,roe = np.ndarray((nx,ny)),np.ndarray((nx,ny)),np.ndarray((nx,ny)),np.ndarray((nx,ny)),np.ndarray((nx,ny))
                for j in range(ny):
                    for i in range(nx):
                        ro[i,j]  = ro_[i][j][0]
                        rou[i,j] = rou_[i][j][0]
                        rov[i,j] = rov_[i][j][0]
                        row[i,j] = row_[i][j][0]
                        roe[i,j] = roe_[i][j][0]
    
        return ro,rou,rov,row,roe 
    
    def read_restart(self):
        """Read and stores restart info for all blocks within the domaine"""
        ind = 0
        restart_dict        = {}
        restart_dict[ind]   = {}
        print("*======")
        print("Reading restart")
        for blc in range(1,self.dict_info["nbloc"]+1):
            if ind==0: restart = self.repo+'/restart_bl'+str(blc)+'.bin'
            print("Restart file is:", restart)
            restart_dict[ind][blc] = {}
            if self.nz[1]==1:
                ro,rou,rov,row,roe = self.read_restart_block(restart,self.nx[blc],self.ny[blc], self.nz[blc])
                if type(ro)==np.ndarray:
                    restart_dict[ind]['uu'] = np.ndarray((self.ny[blc],self.nx[blc]))
                    restart_dict[ind]['vv'] = np.ndarray((self.ny[blc],self.nx[blc]))
                    restart_dict[ind]['ut'] = np.ndarray((self.ny[blc],self.nx[blc]))
                    restart_dict[ind]['prs'] = np.ndarray((self.ny[blc],self.nx[blc]))
                    for j in range(self.ny[blc]):
                        for i in range(self.nx[blc]):
                            restart_dict[ind]['uu'][j][i]  = rou[i][j] / ro[i][j]
                            restart_dict[ind]['ut'][j][i]  = restart_dict[ind]['uu'][j][i] - \
                                self.dict_info["cref"]*self.dict_info["Mref"]
                            restart_dict[ind]['vv'][j][i]  = rov[i][j] / ro[i][j]
                            restart_dict[ind]['prs'][j][i] = (self.gamma-1.)*(roe[i][j] - \
                                                            0.5*(rou[i][j]**2 + rov[i][j]**2) / ro[i][j]) - self.dict_info["Pref"] 
            else:

                if ind==0:
                    ro,rou,rov,row,roe = {},{},{},{},{}
                    ro,rou,rov,row,roe = self.read_restart_block(restart, self.nx[blc], self.ny[blc], self.nz[blc])
                elif ind==1:
                    u0,v0,w0,p0,ro,T0 = {},{},{},{},{},{}
                    u0,v0,w0,p0,ro,T0 = self.read_mean(restart,self.nx[blc], self.ny[blc], self.nz[blc])

                if type(ro)!=np.ndarray:
                    if ind==0: is_restart=False
                    break

                if type(ro)==np.ndarray:
                    # Plan z
                    z_v = self.nz[blc]//2
                    restart_dict[ind][blc]['uu'] = np.ndarray((self.nx[blc], self.ny[blc]))
                    restart_dict[ind][blc]['vv'] = np.ndarray((self.nx[blc], self.ny[blc]))
                    restart_dict[ind][blc]['ww'] = np.ndarray((self.nx[blc], self.ny[blc]))
                    restart_dict[ind][blc]['ut'] = np.ndarray((self.nx[blc], self.ny[blc]))
                    restart_dict[ind][blc]['prs']= np.ndarray((self.nx[blc], self.ny[blc]))
                    if ind==0:
                        for j in range(self.ny[blc]):
                            for i in range(self.nx[blc]):
                                print("rou[i][j]:", rou[i][j])
                                restart_dict[ind][blc]['uu'][i][j]  = rou[i][j][z_v] / ro[i][j][z_v]
                                restart_dict[ind][blc]['ut'][i][j]  = restart_dict[ind][blc]['uu'][i][j] - \
                                    self.dict_info["cref"]*self.dict_info["Mref"]
                                restart_dict[ind][blc]['vv'][i][j]  = rov[i][j][z_v] / ro[i][j][z_v]
                                restart_dict[ind][blc]['ww'][i][j]  = row[i][j][z_v] / ro[i][j][z_v]
                                restart_dict[ind][blc]['prs'][i][j] = (self.gamma-1.)*(roe[i][j][z_v] - \
                                                0.5*(rou[i][j][z_v]**2 + rov[i][j][z_v]**2) / ro[i][j][z_v]) - self.dict_info["Pref"]
                    elif ind==1:
                        for j in range(self.ny[blc]):
                            for i in range(self.nx[blc]):
                                restart_dict[ind][blc]['uu'][i][j]  = u0[i][j][z_v]
                                restart_dict[ind][blc]['ut'][i][j]  = restart_dict[ind][blc]['uu'][i][j] - \
                                                    self.dict_info["cref"]*self.dict_info["Mref"]
                                restart_dict[ind][blc]['vv'][i][j]  = v0[i][j][z_v]
                                restart_dict[ind][blc]['ww'][i][j]  = w0[i][j][z_v]
                                restart_dict[ind][blc]['prs'][i][j] = p0[i][j][z_v] - self.dict_info["Pref"]

                    # Plan y
                    y_v = 1
                    restart_dict[ind][blc]['uu_y'] = np.ndarray((self.nz[blc], self.nx[blc]))
                    restart_dict[ind][blc]['vv_y'] = np.ndarray((self.nz[blc], self.nx[blc]))
                    restart_dict[ind][blc]['ww_y'] = np.ndarray((self.nz[blc], self.nx[blc]))
                    restart_dict[ind][blc]['ut_y'] = np.ndarray((self.nz[blc], self.nx[blc]))
                    restart_dict[ind][blc]['Pp_y'] = np.ndarray((self.nz[blc], self.nx[blc]))
                    if ind==0:
                        for k in range(self.nz[blc]):
                            for i in range(self.nx[blc]):
                                restart_dict[ind][blc]['uu_y'][k][i]  = rou[i][y_v][k] / ro[i][y_v][k]
                                restart_dict[ind][blc]['ut_y'][k][i]  = restart_dict[ind][blc]['uu_y'][k][i] - self.dict_info["cref"]*self.dict_info["Mref"]
                                restart_dict[ind][blc]['vv_y'][k][i]  = rov[i][y_v][k] / ro[i][y_v][k]
                                restart_dict[ind][blc]['ww_y'][k][i]  = row[i][y_v][k] / ro[i][y_v][k]
                                restart_dict[ind][blc]['Pp_y'][k][i] = (self.gamma-1.)*(roe[i][y_v][k] - 0.5*(rou[i][y_v][k]**2 + rov[i][y_v][k]**2) / ro[i][y_v][k]) - self.dict_info["Pref"]
                    elif ind==1:
                        for k in range(self.nz[blc]):
                            for i in range(self.nx[blc]):
                                restart_dict[ind][blc]['uu_y'][k][i] = u0[i][y_v][k]
                                restart_dict[ind][blc]['ut_y'][k][i] = restart_dict[ind][blc]['uu_y'][k][i] - self.dict_info["cref"]*self.dict_info["Mref"]
                                restart_dict[ind][blc]['vv_y'][k][i] = v0[i][y_v][k]
                                restart_dict[ind][blc]['ww_y'][k][i] = w0[i][y_v][k]
                                restart_dict[ind][blc]['Pp_y'][k][i] = p0[i][y_v][k] - self.dict_info["Pref"]

                    # Plan x
                    x_v = 1
                    restart_dict[ind][blc]['uu_x'] = np.ndarray((self.ny[blc], self.nz[blc]))
                    restart_dict[ind][blc]['vv_x'] = np.ndarray((self.ny[blc], self.nz[blc]))
                    restart_dict[ind][blc]['ww_x'] = np.ndarray((self.ny[blc], self.nz[blc]))
                    restart_dict[ind][blc]['ut_x'] = np.ndarray((self.ny[blc], self.nz[blc]))
                    restart_dict[ind][blc]['Pp_x'] = np.ndarray((self.ny[blc], self.nz[blc]))
                    if ind==0:
                        for k in range(nz[blc]):
                            for j in range(ny[blc]):
                                restart_dict[ind][blc]['uu_x'][j][k] = rou[x_v][j][k] / ro[x_v][j][k]
                                restart_dict[ind][blc]['ut_x'][j][k] = restart_dict[ind][blc]['uu_x'][j][k] - self.dict_info["cref"]*self.dict_info["Mref"]
                                restart_dict[ind][blc]['vv_x'][j][k] = rov[x_v][j][k] / ro[x_v][j][k]
                                restart_dict[ind][blc]['ww_x'][j][k] = row[x_v][j][k] / ro[x_v][j][k]
                                restart_dict[ind][blc]['Pp_x'][j][k] = (self.gamma-1.)*(roe[x_v][j][k] - 0.5*(rou[x_v][j][k]**2 + rov[x_v][j][k]**2) / ro[x_v][j][k]) - self.dict_info["Pref"]
                    elif ind==1:
                        for k in range(nz[blc]):
                            for j in range(ny[blc]):
                                restart_dict[ind][blc]['uu_x'][j][k] = u0[x_v][j][k]
                                restart_dict[ind][blc]['ut_x'][j][k] = restart_dict[ind][blc]['uu_x'][j][k] - self.dict_info["cref"]*self.dict_info["Mref"]
                                restart_dict[ind][blc]['vv_x'][j][k] = v0[x_v][j][k]
                                restart_dict[ind][blc]['ww_x'][j][k] = w0[x_v][j][k]
                                restart_dict[ind][blc]['Pp_x'][j][k] = p0[x_v][j][k] - self.dict_info["Pref"]

                else:
                    print("~> Restart file for block %s not found"%(blc))

        return restart_dict

    # ===================
    # Read RFM turbulent:
    # ===================
                
    def read_RFM_turbul(self, 
                        filename):
        """Reads Random Fourier Modes info from a binary file.

        Args:
            filename (str): Name of the binary file containing the RFM data.
    
        Returns:
            tuple: Tuple containing RFM data.
        """
        # option is_check_conv in param_RFM.ini
        try:
            f=open(self.repo+"/"+filename,'r')
            print("Reading %s..."%(self.repo))
        except FileNotFoundError:
            print("File %s not found..."%(self.repo))
            return [-1],[-1],[-1],[-1],[-1],[-1]

        # Reading of the rms of u,v,w,uv,uw,vw
        # ------------------------------------
        arg = np.fromfile(f, dtype=('>f4'), count=1)
        ngy = np.fromfile(f, dtype=('>i4'), count=1)[0]
        #==============================================
        arg = np.fromfile(f, dtype=('>f8'), count=1)
        ngz = np.fromfile(f, dtype=('>i4'), count=1)[0]
        #==============================================
        arg = np.fromfile(f, dtype=('>f8'), count=1)
        yg_ = np.zeros((ngy),dtype='float',order='F')
        yg_ = np.fromfile(f, dtype=('>f8'), count=ngy)
        #==============================================
        arg = np.fromfile(f, dtype=('>f8'), count=1)
        zg_ = np.zeros((ngz),dtype='float',order='F')
        zg_ = np.fromfile(f, dtype=('>f8'), count=ngz)
        #==============================================
        arg = np.fromfile(f, dtype=('>f8'), count=1)
        self.utrms = np.zeros((ngy),dtype='float',order='F')
        self.utrms = np.fromfile(f, dtype=('>f8'), count=ngy)
        #==============================================
        arg = np.fromfile(f, dtype=('>f8'), count=1)
        self.vtrms = np.zeros((ngy),dtype='float',order='F')
        self.vtrms = np.fromfile(f, dtype=('>f8'), count=ngy)
        #==============================================
        arg = np.fromfile(f, dtype=('>f8'), count=1)
        self.wtrms = np.zeros((ngy),dtype='float',order='F')
        self.wtrms = np.fromfile(f, dtype=('>f8'), count=ngy)
        #==============================================
        arg = np.fromfile(f, dtype=('>f8'), count=1)
        self.uvtrms = np.zeros((ngy),dtype='float',order='F')
        self.uvtrms = np.fromfile(f, dtype=('>f8'), count=ngy)
        #==============================================
        arg = np.fromfile(f, dtype=('>f8'), count=1)
        self.uwtrms = np.zeros((ngy),dtype='float',order='F')
        self.uwtrms = np.fromfile(f, dtype=('>f8'), count=ngy)
        #==============================================
        arg = np.fromfile(f, dtype=('>f8'), count=1)
        self.vwtrms = np.zeros((ngy),dtype='float',order='F')
        self.vwtrms = np.fromfile(f, dtype=('>f8'), count=ngy)
        return self.utrms, self.vtrms, self.wtrms, self.uvtrms, self.uwtrms, self.vwtrms
    
    # ====================
    # Read stats for STBL:
    # ====================
    
    def read_stats_stbl(self, 
                        bloc, 
                        verbose=True, 
                        is_st1=False, 
                        is_st_wn=False):
        """Reads statistics data for STBL configuration.

        Args:
            bloc (list): List of block indices to read statistics for.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            is_st1 (bool, optional): Whether to read only the first stats1 file. Defaults to False.
            is_st_wn (bool, optional): Whether to use stats1_wn.bin file. Defaults to False.

        Returns:
            dict, dict: Dictionaries containing statistics data for stats1 and stats2 files respectively.
            """
    # Reading stats1_bl*.bin and stats2_bl*.bin files for STBL configuration
        repertory = self.repo
        nbloc = self.dict_info["nbloc"]

        self._stats1, self._stats2 = {}, {}
        for i in range(1, nbloc + 1):
            self._stats1[i] = {}
            self._stats2[i] = {}

        print("*======") 
        
        # Reading of the stats
        # --------------------
        # Reading of stats1_bl.bin
        for i in bloc:
            nx = self.nx[i]
            ny = self.ny[i]
            if is_st_wn:
                file_stats1 = repertory + '/stats1_wn.bin'
            else:
                file_stats1 = repertory + '/stats1_bl' + str(i) + '.bin'
            plane = self.read_plane_block(filename=file_stats1, n1=nx, n2=ny, nvar=23, verbose=verbose)[0]
            if plane is not None:
                print(f"Reading stats from block {i} in files {file_stats1} hehe")
                for j, var_name in enumerate([
                    'rho', 'u', 'v', 'w', 'p', 'T',
                    'rhou', 'rhov', 'rhow', 'rhoe', 'rho^2',
                    'u2', 'v2', 'w2', 'uv', 'uw', 'vw',
                    'vT', 'p2', 'T2', 'mu', 'div', 'div^2'
                ]):

                    value = plane[f'var{j + 1}']
                    # Check if data contains field or not
                    if value != None:
                        self._stats1[i][var_name] = value if isinstance(value, list) and len(value) > 1 else value[0]
            else:
                self._stats1 = None
                print("plane is none!")
        
        # Reading of stats2_bl.bin
        i = 0
        if not is_st1:
            for i in bloc:
                nx = self.nx[i]
                ny = self.ny[i]
                if is_st_wn:
                    file_stats2 = repertory + '/stats2_wn.bin'
                else:
                    file_stats2 = repertory + '/stats2_bl' + str(i) + '.bin'
                plane = self.read_plane_block(filename=file_stats2, n1=nx, n2=ny, nvar=154, verbose=verbose)[0]
                
                if plane is not None:
                    print(f"Reading stats from block {i} in files {file_stats2}")
                    #print(f"we are now at {j+1}th key")
                    for j, var_name in enumerate([
                        'e', 'h', 'c', 's', 'M', 'kt', 'g', 'cok', 'cp', 'cv',
                        'pr', 'eck', 'rho*dux', 'rho*duy', 'rho*duz', 'rho*dvx', 'rho*dvy', 'rho*dvz',
                        'rho*dwx', 'rho*dwy', 'rho*dwz', 'p*div', 'rho*div', 'b1', 'b2', 'b3', 'rho*T',
                        'u*T', 'v*T', 'e^2', 'h^2', 'c^2', 's^2', 'Mt^2', 'g^2', 'mu^2', 'cok^2', 'cv^2',
                        'cp^2', 'pr^2', 'eck^2', 'p*u', 'p*v', 's*u', 's*v', 'p*rho', 'h*rho', 'T*p', 'p*s',
                        'T*s', 'rho*s', 'g*rho', 'g*p', 'g*s', 'g*T', 'g*u', 'g*v', 'p*dux', 'p*dvy', 'p*dwz',
                        'p*duy', 'p*dvx', 'rho*u^2' , 'dux^2', 'duy^2', 'duz^2', 'dvx^2', 'dvy^2', 'dvz^2', 'dwx^2',
                        'dwy^2', 'dwz^2', 'b1^2', 'b2^2', 'b3^2', 'rho*b1', 'rho*b2', 'rho*b3', 'rho*u^2', 'rho*v^2', 
                        'rho*w^2', 'rho*T^2', 'rho*b1^2', 'rho*b2^2', 'rho*b3^2', 'rho*u*v', 'rho*u*w', 'rho*v*w',
                        'rho*v*T', 'rho*u^2*v', 'rho*v^2*v', 'rho*w^2*v', 'rho*v^2*u', 'rho*dux^2', 'rho*dvy^2',
                        'rho*dwz^2', 'rho*duy*dvx', 'rho*duz*dwx', 'rho*dvz*dwy', 'u^3', 'p^3', 'u^4', 'p^4', 'Frhou',
                        'Frhov', 'Frhow', 'Grhov', 'Grhow', 'Hrhow', 'Frhov*u', 'Frhou*u', 'Frhov*v', 'Frhow*w', 
                        'Grhov*u', 'Grhov*v', 'Grhow*w', 'Frhou*dux', 'Frhou*dvx', 'Frhov*dux', 'Frhov*duy', 'Frhov*dvx',
                        'Frhov*dvy', 'Frhow*duz', 'Frhow*dvz', 'Frhow*dwx', 'Grhov*duy', 'Grhov*dvy', 'Grhow*duz',
                        'Grhow*dvz', 'Grhow*dwy', 'Hrhow*dwz', 'la*dTx', 'la*dTy', 'la*dTz', 'h*u', 'h*v', 'h*w',
                        'rho*h*u', 'rho*h*v', 'rho*h*w', 'rho*u^3', 'rho*v^3', 'rho*w^3', 'rho*w^2*u', 'mut'
                    ]):
                        value = plane.get(f'var{j + 1}')
                        if value is not None and value:
                            self._stats2[i][var_name] = value if isinstance(value, list) and len(value) > 1 else value[0]
                        else:
                            print(f"Variable {var_name} not found or is None")
                else:
                    self._stats2 = None
        print("*======") 



    def read_vkm(self):
        """Reads vkm.bin file.
        
        Returns:
            tuple: Tuple containing data from vkm.bin file.
        """
        sens = '>'
        if self.is_little_endian: sens = '<'
        try:
            print("*======")
            print("Reading vkm.bin")
            f = open(self.repo+'/vkm.bin','r')
            arg     = np.fromfile(f, dtype=(sens+'i4'), count=1)
            Nmodes  = np.fromfile(f, dtype=(sens+'i4'), count=1)[0]
            arg     = np.fromfile(f, dtype=(sens+'i8'), count=1)
            xkn     = np.fromfile(f, dtype=(sens+'f8'), count=Nmodes)
            arg     = np.fromfile(f, dtype=(sens+'i8'), count=1)
            dxkn    = np.fromfile(f, dtype=(sens+'f8'), count=Nmodes)
            arg     = np.fromfile(f, dtype=(sens+'i8'), count=1)
            vkm     = np.fromfile(f, dtype=(sens+'f8'), count=Nmodes)
        except FileNotFoundError:
            print("File not found to read vkm")
            res_vkm = False
        return xkn,vkm
