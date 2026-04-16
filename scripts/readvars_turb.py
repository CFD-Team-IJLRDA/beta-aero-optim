import numpy as np
import scipy.integrate as si
import scipy.stats as ss
import statsmodels.api as sa
from collections import OrderedDict
import total_prsv
import total_pfg
import post_shock_prsv
import post_shock_pfg

def d():
    x = OrderedDict()
    return x
# endian = 'big'
endian = 'small'

# Read individual grids
#######################

def read_grid(file_input,is_curv,endian):
    # Reading of the grid
    # -------------------
    print("Reading grid...")

    i4_dtype = np.dtype('i4')
    i8_dtype = np.dtype('i8')
    f4_dtype = np.dtype('f4')
    f8_dtype = np.dtype('f8')
    if endian=='big':
        i4_dtype = np.dtype('>i4')
        f8_dtype = np.dtype('>f8')


    f = open(file_input,'r')
    arg = np.fromfile(f, dtype=i4_dtype, count=1)
    nx  = np.fromfile(f, dtype=i4_dtype, count=1)[0]
    #===============================================================================
    arg = np.fromfile(f, dtype=i8_dtype, count=1)
    ny  = np.fromfile(f, dtype=i4_dtype, count=1)[0]
    #===============================================================================
    arg = np.fromfile(f, dtype=i8_dtype, count=1)
    nz  = np.fromfile(f, dtype=i4_dtype, count=1)[0]
    #===============================================================================
    arg = np.fromfile(f, dtype=i8_dtype, count=1)
    is_curv3 = False
    if is_curv3:
        x = np.zeros((nx,ny,nz),dtype=f8_dtype,order='F')
        for k in range(nz):
            for j in range(ny):
                x[:,j,k] = np.fromfile(f, dtype=f8_dtype, count=nx)
        #===============================================================================
        arg = np.fromfile(f, dtype=i8_dtype, count=1)
        y = np.zeros((nx,ny,nz),dtype=f8_dtype,order='F')
        for k in range(nz):
            for j in range(ny):
                y[:,j,k] = np.fromfile(f, dtype=f8_dtype, count=nx)
        #===============================================================================
        arg = np.fromfile(f, dtype=i8_dtype, count=1)
        z = np.zeros((nx,ny,nz),dtype=f8_dtype,order='F')
        for k in range(nz):
            for j in range(ny):
                z[:,j,k] = np.fromfile(f, dtype=f8_dtype, count=nx)
        #===============================================================================
    elif is_curv:
        x = np.zeros((nx,ny),dtype=f8_dtype,order='F')
        for j in range(ny):
            x[:,j] = np.fromfile(f, dtype=f8_dtype, count=nx)
        #===============================================================================
        arg = np.fromfile(f, dtype=i8_dtype, count=1)
        y = np.zeros((nx,ny),dtype=f8_dtype,order='F')
        for j in range(ny):
            y[:,j] = np.fromfile(f, dtype=f8_dtype, count=nx)
        #===============================================================================
        arg = np.fromfile(f, dtype=i8_dtype, count=1)
        z = np.fromfile(f, dtype=f8_dtype, count=nz)
        #===============================================================================
    else:
        x = np.zeros((nx),dtype='float',order='F')
        x = np.fromfile(f, dtype=f8_dtype, count=nx)
        #===============================================================================
        arg = np.fromfile(f, dtype=i8_dtype, count=1)
        y = np.zeros((ny),dtype='float',order='F')
        y = np.fromfile(f, dtype=f8_dtype, count=ny)
        #===============================================================================
        arg = np.fromfile(f, dtype=i8_dtype, count=1)
        z = np.zeros((ny),dtype='float',order='F')
        z = np.fromfile(f, dtype=f8_dtype, count=nz)
        #===============================================================================
        x,y = np.meshgrid(x,y)

    return nx,ny,nz,x,y,z


# Read restart files
####################

def read_restart(file_input,nx,ny,nz,is_2D,is_RANS):
    print("Reading restart...")
    f=open(file_input,'r')
    ro  = np.fromfile(f, dtype=('<f8'), count=nx*ny*nz).reshape((nx,ny,nz), order='F')
    rou = np.fromfile(f, dtype=('<f8'), count=nx*ny*nz).reshape((nx,ny,nz), order='F')
    rov = np.fromfile(f, dtype=('<f8'), count=nx*ny*nz).reshape((nx,ny,nz), order='F')
    row = np.fromfile(f, dtype=('<f8'), count=nx*ny*nz).reshape((nx,ny,nz), order='F')
    roe = np.fromfile(f, dtype=('<f8'), count=nx*ny*nz).reshape((nx,ny,nz), order='F')
    if is_RANS:
        nutil = np.fromfile(f, dtype=('<f8'), count=nx*ny*nz).reshape((nx,ny,nz), order='F')
    f.close()

    # w = row/ro
    # maxw = np.where(w==w.max())
    # minw = np.where(w==w.min())
    # print(minw,w[minw],maxw,w[maxw])

    if is_2D:
        ro_,rou_,rov_,row_,roe_ = ro,rou,rov,row,roe
        ro,rou,rov,row,roe = np.ndarray((nx,ny)),np.ndarray((nx,ny)),np.ndarray((nx,ny)),np.ndarray((nx,ny)),np.ndarray((nx,ny))
        for j in range(ny):
            for i in range(nx):
                ro[i][j]  = ro_[i][j][0]
                rou[i][j] = rou_[i][j][0]
                rov[i][j] = rov_[i][j][0]
                row[i][j] = row_[i][j][0]
                roe[i][j] = roe_[i][j][0]


        if is_RANS:
           nutil_ = nutil
           nutil  = np.ndarray((nx,ny))
           for j in range(ny):
               for i in range(nx):
                   nutil[i][j] = nutil_[i][j][0]
    if is_RANS:
        return ro,rou,rov,row,roe,nutil
    else:
        return ro,rou,rov,row,roe


# Timestamps
############

def read_time(file_input):
    print("Reading time.ini...")
    dict_time = {"timestamp" : [],
                 "ite": [],
                 "time": []}
    f = open(file_input,'r')
    lines = f.readlines()[1:]
    for line in lines:
        line = line.split()
        dict_time["timestamp"].append(line[0])
        dict_time["ite"].append(int(line[2]))
        dict_time["time"].append(float(line[4]))
    return dict_time


# Read simulation info
######################

def read_info(dir_data):
    dict_info = {}

    f = open(dir_data+'info.ini','r')
    lines = f.readlines()
    dict_info["nbloc"]  = int(lines[0].split()[4])
    dict_info["is_curv"]= lines[0].split()[5]
    for ind in range(dict_info["nbloc"]):
        dict_info["nx_bl"+str(ind+1)] = int(lines[1+ind].split()[5])
        dict_info["ny_bl"+str(ind+1)] = int(lines[1+ind].split()[6])
        dict_info["nz_bl"+str(ind+1)] = int(lines[1+ind].split()[7])
    dict_info["etot0"]  = float(lines[2+ind].split()[3])
    dict_info["mgtot0"] = float(lines[2+ind].split()[4])
    dict_info["xmin"]   = float(lines[3+ind].split()[4])
    dict_info["ymin"]   = float(lines[3+ind].split()[5])
    dict_info["zmin"]   = float(lines[3+ind].split()[6])
    dict_info["xmax"]   = float(lines[4+ind].split()[4])
    dict_info["ymax"]   = float(lines[4+ind].split()[5])
    dict_info["zmax"]   = float(lines[4+ind].split()[6])
    dict_info["Mref"]   = float(lines[5+ind].split()[3])
    dict_info["Reref"]  = float(lines[5+ind].split()[4])
    dict_info["Mupref"] = float(lines[6+ind].split()[3])
    dict_info["Muref"]  = float(lines[6+ind].split()[4])
    dict_info["Roref"]  = float(lines[7+ind].split()[4])
    dict_info["Pref"]   = float(lines[7+ind].split()[5])
    dict_info["Tref"]   = float(lines[7+ind].split()[6])
    dict_info["Uref"]   = float(lines[8+ind].split()[4])
    dict_info["cref"]   = float(lines[8+ind].split()[5])
    dict_info["Tscale"] = float(lines[8+ind].split()[6])
    dict_info["L_ref"]  = dict_info['Reref']*dict_info['Muref']/ \
                          dict_info['Roref']/(dict_info['Uref'])
    dict_info["dt"]     = float(lines[9+ind].split()[4])
    
    return dict_info


# Read planes
#############

def read_planes(file_input,chkpnt_nb,wall_plane,anim):
    print("Reading plane...")
    bl = file_input.split('_')[-1].split('.')[0]
    dir_data  = file_input.split("plane")[0]
    dict_info = read_info(dir_data)
    nx,ny,nz = dict_info[f'nx_{bl}'], \
               dict_info[f'ny_{bl}'], \
               dict_info[f'nz_{bl}']
    u_in   = dict_info['Uref']
    L_ref  = dict_info['L_ref']
    mu_ref = dict_info['Muref']

    # Find nb of planes written in file
    f = open(file_input,'rb')
    f8_dtype = np.dtype('f8')
    data = np.fromfile(f, dtype=f8_dtype, count=-1)
    plane = d()
    planes = d()

    # Plane normal to z-axis
    if not wall_plane:
        var_names = ['u','v','p','T','rho','M','cfl_i','cfl_j','s','h','vort','nutil','St1','St2','St3','mut','Sterm','dtlocal']
        # var_names = ['u','v','w','p','T','rho','div','M','vort','cfl_i','cfl_j']
        # var_names = ['u','v','p','T','rho','M','cfl_i','cfl_j']
        # var_names = ['u','v','p','rho']
        # var_names = ['u','v','w','p','T','rho']
        planes['nb_chkpnts'] = data.size//(nx*ny)//len(var_names)
        print('nb_chkpnts =',planes['nb_chkpnts'])
        chkpnt_nb = len(var_names)*(chkpnt_nb-1)
        for nvar,var in enumerate(var_names):
            plane[var] = data[nx*ny*(nvar+chkpnt_nb):nx*ny*(nvar+1+chkpnt_nb)]\
                         .reshape((nx,ny),order='F')
            if var=='mut':
                plane[var] = plane[var]/mu_ref
            if var=='vort':
                plane[var] = abs(plane[var]*L_ref/u_in)
            if var=='St1':
                ind_max = np.unravel_index(np.argmax(plane[var],axis=None),\
                      plane[var].shape)
                print('Max St1 indices',ind_max,plane[var].max())

        # Animation
        if anim:
            for i in range(planes['nb_chkpnts']):
                i+=1
                planes[f'{i}'] = d()
                chkpnt_nb = len(var_names)*(i-1)
                for nvar,var in enumerate(var_names):
                    planes[f'{i}'][var] = data[nx*ny*(nvar+chkpnt_nb):nx*ny*(nvar+1+chkpnt_nb)]\
                                          .reshape((nx,ny),order='F')
                    if var=='mut':
                        planes[f'{i}'][var] = planes[f'{i}'][var]/mu_ref
                    planes[f'{i}'][f'vmax_{var}'] = planes[f'{i}'][var].max()
                    planes[f'{i}'][f'vmin_{var}'] = planes[f'{i}'][var].min()

    # Wall surface normal to j-dir
    else:
        # var_names = ['p','rho','Frhov','Frhou','Frhow','Grhov','Grhow','Hrhow']
        var_names = ['p','rho','Frhov','Frhou','Grhov']
        planes['nb_chkpnts'] = data.size//(nx*nz)//len(var_names)
        print(planes['nb_chkpnts'])

        chkpnt_nb = len(var_names)*(chkpnt_nb-1)
        for nvar,var in enumerate(var_names):
            plane[var] = data[nx*nz*(nvar+chkpnt_nb):nx*nz*(nvar+1+chkpnt_nb)]\
                         .reshape((nx,nz),order='F')
            if var=='vort':
                plane[var] = abs(plane[var]*L_ref/u_in)

        # Animation
        if anim:
            for i in range(planes['nb_chkpnts']):
                i+=1
                planes[f'{i}'] = d()
                chkpnt_nb = len(var_names)*(i-1)
                for nvar,var in enumerate(var_names):
                    planes[f'{i}'][var] = data[nx*nz*(nvar+chkpnt_nb):nx*nz*(nvar+1+chkpnt_nb)]\
                                          .reshape((nx,nz),order='F')
                    planes[f'{i}'][f'vmax_{var}'] = planes[f'{i}'][var].max()

    return plane,var_names,planes


# Extract sensor from plane
###########################

def sensor_plane(dir_data,var,plane_nb,bl,ind_i,ind_j,wall_plane):
    plane_nb = str(plane_nb).rjust(3,'0')
    plane_file = dir_data+f'plane_{plane_nb}_sol_bl{bl}.bin'
    plane,var_names,planes = read_planes(plane_file,1,wall_plane,True)
    nb_chkpnts = planes['nb_chkpnts']
    sensor = np.zeros((nb_chkpnts))
    for i in range(nb_chkpnts):
        sensor[i] = (planes[f'{i+1}'][var][ind_i,ind_j]+planes[f'{i+1}'][var][ind_i+1,ind_j]+\
                     planes[f'{i+1}'][var][ind_i-1,ind_j]+planes[f'{i+1}'][var][ind_i,ind_j+1]+\
                     planes[f'{i+1}'][var][ind_i,ind_j-1])/5

    plane_file2 = dir_data+f'plane_{plane_nb}_sol_bl{2}.bin'
    plane2,var_names,planes2 = read_planes(plane_file2,1,wall_plane,True)
    nb_chkpnts = planes['nb_chkpnts']
    sensor2 = np.zeros((nb_chkpnts))
    ind_i,ind_j = 150,200
    for i in range(nb_chkpnts):
        sensor2[i] = (planes2[f'{i+1}'][var][ind_i,ind_j]+planes2[f'{i+1}'][var][ind_i+1,ind_j]+\
                     planes2[f'{i+1}'][var][ind_i-1,ind_j]+planes2[f'{i+1}'][var][ind_i,ind_j+1]+\
                     planes2[f'{i+1}'][var][ind_i,ind_j-1])/5
        # sensor[i] = planes[f'{i+1}'][var][ind_i,ind_j]
    return sensor,sensor2,nb_chkpnts
    


# Write plane files
###################

def write_planes(dir_data1,dir_data2,plane_nb):

    plane_nb = str(plane_nb).rjust(3,'0')
    n_bl = 12
    for bl in range(n_bl):
        bl+=1
        file1 = dir_data1+f'plane_{plane_nb}_sol_bl{bl}.bin'
        file2 = dir_data2+f'plane_{plane_nb}_sol_bl{bl}.bin'

        f1 = open(file1,'rb')
        f2 = open(file2,'rb')
        f8_dtype = np.dtype('f8')
        data1 = np.fromfile(f1, dtype=f8_dtype, count=-1)
        data2 = np.fromfile(f2, dtype=f8_dtype, count=-1)

        data_bin = np.hstack((data1,data2))

        data_bin.tofile(f'{dir_data2}/combined_planes/plane_{plane_nb}_sol_bl{bl}.bin')
    print(f'File written in {dir_data2}/combined_planes/')


# Read aero coeffs
##################

def read_coeffs(time_only,file_input,iter_avg):
    print("Reading aerodynamic coefficients...")
    f = open(file_input,'r')
    data = np.loadtxt(f,dtype='float')

    niter,time,tstar,cl,cd = \
            data[:,0],data[:,1],data[:,2],data[:,3],data[:,4]
    f.close()

    if time_only:
        return niter,tstar

    niter = niter-niter[0]+1
    every = int(niter[1]-niter[0])
    niter_avg = (niter[-1]-iter_avg)//every
    iter_avg = int(iter_avg)//every
    cl_avg,cd_avg = np.mean(cl[iter_avg:]), \
                    np.mean(cd[iter_avg:])
    cl_rms,cd_rms = np.sqrt(np.mean((cl[iter_avg:]-cl_avg)**2)), \
                    np.sqrt(np.mean((cd[iter_avg:]-cd_avg)**2))

    return niter,every,time,tstar,cl,cd,cl_avg,cd_avg,cl_rms,cd_rms


# Read nusselt evolution
########################

def read_nusselt(file_input,iter_avg):
    print("Reading Nusselt number evolution...")
    f = open(file_input,'r')
    data = np.loadtxt(f,dtype='float')

    niter,time,tstar,nus = \
            data[:,0],data[:,1],data[:,2],data[:,3]
    f.close()
    
    niter_avg = niter[-1]-iter_avg
    nus_avg = sum(nus[iter_avg:])/niter_avg
    nus_rms = np.sqrt(sum((nus[iter_avg:]-nus_avg)**2)/niter_avg)

    return niter,time,tstar,nus,nus_avg,nus_rms


# Read residuals
################

def read_residuals(file_input,is_RANS):
    print("Reading residuals...")
    is_2D = True
    if is_2D:
        res = {'nn': [], 'Rho': [], 'Rhou': [], 'Rhov': [], 'Rhoe': []}
    if is_RANS:
        res = {'nn': [], 'Rho': [], 'Rhou': [], 'Rhov': [], 'Rhoe': [], 'Rhonutil': []}
    else:
        res = {'nn': [], 'Rho': [], 'Rhou': [], 'Rhov': [], 'Rhow': [], 'Rhoe': []}
    # Reading of the residuals
    # ------------------------
    f = open(file_input,'r')
    i4_dtype = np.dtype('i4')
    i8_dtype = np.dtype('i8')
    f4_dtype = np.dtype('f4')
    f8_dtype = np.dtype('f8')

    while True:
        try:
            arg = np.fromfile(f, dtype=f8_dtype, count=1)
            res['nn'].append(np.fromfile(f, dtype=i8_dtype, count=1)[0])
            #==============================================
            arg = np.fromfile(f, dtype=f8_dtype, count=1)
            res['Rho'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
            #==============================================
            arg = np.fromfile(f, dtype=f8_dtype, count=1)
            res['Rhou'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
            #==============================================
            arg = np.fromfile(f, dtype=f8_dtype, count=1)
            res['Rhov'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
            #==============================================
            if is_2D:
                arg = np.fromfile(f, dtype=f8_dtype, count=1)
                res['Rhoe'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
                if is_RANS:
                    arg = np.fromfile(f, dtype=f8_dtype, count=1)
                    res['Rhonutil'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
            else:
                arg = np.fromfile(f, dtype=f8_dtype, count=1)
                res['Rhoe'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
                arg = np.fromfile(f, dtype=f8_dtype, count=1)
                res['Rhoe'].append(np.fromfile(f, dtype=f8_dtype, count=1)[0])
            arg = np.fromfile(f, dtype=f4_dtype, count=1)
        except IndexError:
            break
    return res


# Read statistics file
######################

def read_stats(file_input,geom,mesh,is_RANS,nx,ny):
    print("Reading stats...")
    stats = d()
    if file_input[-14:].split('_')[0][-1]=='1':
        var_list = ['rho','u','v','w','p','T','rhou','rhov','rhow','rhoe', \
                    'rho**2','uu','vv','ww','uv','uw','vw','vT','p**2','T**2',\
                    'mu','divloc','divloc**2']
    else:
        if mesh=='C_air_musica3D':
            var_list = ['e','h','c','s','M','0.5*q','g','la','cp','cv', \
                        'prr','eck','rho*dux','rho*duy','rho*duz','rho*dvx','rho*dvy', \
                        'rho*dvz','rho*dwx','rho*dwy','rho*dwz','p*div','rho*div','b1', \
                        'b2','b3','rhoT','uT','vT','e**2','h**2','c**2','s**2', \
                        'qq/cc2','g**2','mu**2','la**2','cv**2','cp**2','prr**2','eck**2', \
                        'p*u','p*v','s*u','s*v','p*rho','h*rho','T*p','p*s','T*s','rho*s', \
                        'g*rho','g*p','g*s','g*T','g*u','g*v','p*dux','p*dvy','p*dwz', \
                        'p*duy','p*dvx','rho*div**2','dux**2','duy**2','duz**2','dvx**2', \
                        'dvy**2','dvz**2','dwx**2','dwy**2','dwz**2','b1**2','b2**2','b3**2', \
                        'rho*b1','rho*b2','rho*b3','rho*uu','rho*vv','rho*ww', \
                        'rho*T**2','rho*b1**2','rho*b2**2','rho*b3**2','rho*uv','rho*uw', \
                        'rho*vw','rho*vT','rho*u**2*v','rho*v**3','rho*w**2*v','rho*v**2*u', \
                        'rho*dux**2','rho*dvy**2','rho*dwz**2','rho*duy*dvx','rho*duz*dwx', \
                        'rho*dvz*dwy','u**3','p**3','u**4','p**4','Frhou','Frhov','Frhow', \
                        'Grhov','Grhow','Hrhow','Frhovu','Frhouu','Frhovv','Frhoww', \
                        'Grhovu','Grhovv','Grhoww','Frhou_dux','Frhou_dvx','Frhov_dux', \
                        'Frhov_duy','Frhov_dvx','Frhov_dvy','Frhow_duz','Frhow_dvz', \
                        'Frhow_dwx','Grhov_duy','Grhov_dvy','Grhow_duz','Grhow_dvz', \
                        'Grhow_dwy','Hrhow_dwz','la*dTx','la*dTy','la*dTz', \
                        'h*u','h*v','h*w','rho*h*u','rho*h*v','rho*h*w','rho*h*u','rho*s*v',\
                        'rho*s*w','rho*u**3','rho*v**3',\
                        'rho*w**3','rho*w**2*u','h0','e0','s0','T0','p0','rh0','mut']
            
#         if mesh=='C_refined':
#             var_list = ['e','h','c','s','M','0.5*q','g','la','cp','cv', \
#                         'prr','eck','rho*dux','rho*duy','rho*duz','rho*dvx','rho*dvy', \
#                         'rho*dvz','rho*dwx','rho*dwy','rho*dwz','p*div','rho*div','b1', \
#                         'b2','b3','rhoT','uT','vT','e**2','h**2','c**2','s**2', \
#                         'qq/cc2','g**2','mu**2','la**2','cv**2','cp**2','prr**2','eck**2', \
#                         'p*u','p*v','s*u','s*v','p*rho','h*rho','T*p','p*s','T*s','rho*s', \
#                         'g*rho','g*p','g*s','g*T','g*u','g*v','p*dux','p*dvy','p*dwz', \
#                         'p*duy','p*dvx','rho*div**2','rho*dux**2','rho*duy**2','rho*duz**2','rho*dvx**2', \
#                         'rho*dvy**2','rho*dvz**2','rho*dwx**2','rho*dwy**2','rho*dwz**2','b1**2','b2**2','b3**2', \
#                         'rho*b1','rho*b2','rho*b3','rho*uu','rho*vv','rho*ww', \
#                         'rho*T**2','rho*b1**2','rho*b2**2','rho*b3**2','rho*uv','rho*uw', \
#                         'rho*vw','rho*vT','rho*u**2*v','rho*v**3','rho*w**2*v','rho*v**2*u', \
#                         'rho*dux**2','rho*dvy**2','rho*dwz**2','rho*duy*dvx','rho*duz*dwx', \
#                         'rho*dvz*dwy','u**3','p**3','u**4','p**4','Frhou','Frhov','Frhow', \
#                         'Grhov','Grhow','Hrhow','Frhovu','Frhouu','Frhovv','Frhoww', \
#                         'Grhovu','Grhovv','Grhoww','Frhou_dux','Frhou_dvx','Frhov_dux', \
#                         'Frhov_duy','Frhov_dvx','Frhov_dvy','Frhow_duz','Frhow_dvz', \
#                         'Frhow_dwx','Grhov_duy','Grhov_dvy','Grhow_duz','Grhow_dvz', \
#                         'Grhow_dwy','Hrhow_dwz','la*dTx','la*dTy','la*dTz', \
#                         'h*u','h*v','h*w','rho*h*u','rho*h*v','rho*h*w','rho*h*u','rho*s*v',\
#                         'rho*s*w','rho*u**3','rho*v**3',\
#                         'rho*w**3','rho*w**2*u','h0','e0','s0','T0','p0','rh0']
            
#         elif mesh=='F_r134a_PR32':
#             var_list = ['e','h','c','s','M','0.5*q','g','la','cp','cv', \
#                         'prr','eck','rho*dux','rho*duy','rho*duz','rho*dvx','rho*dvy', \
#                         'rho*dvz','rho*dwx','rho*dwy','rho*dwz','p*div','rho*div','b1', \
#                         'b2','b3','rhoT','uT','vT','e**2','h**2','c**2','s**2', \
#                         'qq/cc2','g**2','mu**2','la**2','cv**2','cp**2','prr**2','eck**2', \
#                         'p*u','p*v','s*u','s*v','p*rho','h*rho','T*p','p*s','T*s','rho*s', \
#                         'g*rho','g*p','g*s','g*T','g*u','g*v','p*dux','p*dvy','p*dwz', \
#                         'p*duy','p*dvx','rho*div**2','rho*dux**2','rho*duy**2','rho*duz**2','rho*dvx**2', \
#                         'rho*dvy**2','rho*dvz**2','rho*dwx**2','rho*dwy**2','rho*dwz**2','b1**2','b2**2','b3**2', \
#                         'rho*b1','rho*b2','rho*b3','rho*uu','rho*vv','rho*ww', \
#                         'rho*T**2','rho*b1**2','rho*b2**2','rho*b3**2','rho*uv','rho*uw', \
#                         'rho*vw','rho*vT','rho*u**2*v','rho*v**3','rho*w**2*v','rho*v**2*u', \
#                         'rho*dux**2','rho*dvy**2','rho*dwz**2','rho*duy*dvx','rho*duz*dwx', \
#                         'rho*dvz*dwy','u**3','p**3','u**4','p**4','Frhou','Frhov','Frhow', \
#                         'Grhov','Grhow','Hrhow','Frhovu','Frhouu','Frhovv','Frhoww', \
#                         'Grhovu','Grhovv','Grhoww','Frhou_dux','Frhou_dvx','Frhov_dux', \
#                         'Frhov_duy','Frhov_dvx','Frhov_dvy','Frhow_duz','Frhow_dvz', \
#                         'Frhow_dwx','Grhov_duy','Grhov_dvy','Grhow_duz','Grhow_dvz', \
#                         'Grhow_dwy','Hrhow_dwz','la*dTx','la*dTy','la*dTz', \
#                         'h*u','h*v','h*w','rho*h*u','rho*h*v','rho*h*w','rho*h*u','rho*s*v',\
#                         'rho*s*w','rho*u**3','rho*v**3',\
#                         'rho*w**3','rho*w**2*u','h0','e0','s0','T0','p0','rh0']
            
        else:
            var_list = ['e','h','c','s','M','0.5*q','g','la','cp','cv', \
                        'prr','eck','rho*dux','rho*duy','rho*duz','rho*dvx','rho*dvy', \
                        'rho*dvz','rho*dwx','rho*dwy','rho*dwz','p*div','rho*div','b1', \
                        'b2','b3','rhoT','uT','vT','e**2','h**2','c**2','s**2', \
                        'qq/cc2','g**2','mu**2','la**2','cv**2','cp**2','prr**2','eck**2', \
                        'p*u','p*v','s*u','s*v','p*rho','h*rho','T*p','p*s','T*s','rho*s', \
                        'g*rho','g*p','g*s','g*T','g*u','g*v','p*dux','p*dvy','p*dwz', \
                        'p*duy','p*dvx','rho*div**2','rho*dux**2','rho*duy**2','rho*duz**2','rho*dvx**2', \
                        'rho*dvy**2','rho*dvz**2','rho*dwx**2','rho*dwy**2','rho*dwz**2','b1**2','b2**2','b3**2', \
                        'rho*b1','rho*b2','rho*b3','rho*uu','rho*vv','rho*ww', \
                        'rho*T**2','rho*b1**2','rho*b2**2','rho*b3**2','rho*uv','rho*uw', \
                        'rho*vw','rho*vT','rho*u**2*v','rho*v**3','rho*w**2*v','rho*v**2*u', \
                        'rho*dux**2','rho*dvy**2','rho*dwz**2','rho*duy*dvx','rho*duz*dwx', \
                        'rho*dvz*dwy','u**3','p**3','u**4','p**4','Frhou','Frhov','Frhow', \
                        'Grhov','Grhow','Hrhow','Frhovu','Frhouu','Frhovv','Frhoww', \
                        'Grhovu','Grhovv','Grhoww','Frhou_dux','Frhou_dvx','Frhov_dux', \
                        'Frhov_duy','Frhov_dvx','Frhov_dvy','Frhow_duz','Frhow_dvz', \
                        'Frhow_dwx','Grhov_duy','Grhov_dvy','Grhow_duz','Grhow_dvz', \
                        'Grhow_dwy','Hrhow_dwz','la*dTx','la*dTy','la*dTz', \
                        'h*u','h*v','h*w','rho*h*u','rho*h*v','rho*h*w','rho*u**3','rho*v**3',\
                        'rho*w**3','rho*w**2*u']

    f = open(file_input,'rb')
    dtype = np.dtype('f8')
    for var in var_list:
        stats[var] = np.fromfile(f,dtype=dtype,count=nx*ny).reshape((nx,ny),order='F')
    f.close()

    return stats


# Shortcut for dicts
####################

def extr_dict(is_RANS,stats,bl,nx,ny,dir_data,mesh,data):

    stats_file = 'stats{0}_bl{1}.bin'.format(stats,bl)
    # Store in dict
    bl_data = read_stats(dir_data+stats_file,'turb',mesh,is_RANS,nx,ny)
    data[f'block_{bl}'] = d()

    return bl_data,data


# Extract data from stats (musicaa)
#########################

def extr_stats(dir_data,var,is_curv,is_RANS,stream,contours,xz,yz,Lx,Cp,Prms, \
               Cf,Nu,yp,BL,BL_scal,k,sl):

    # General information
    var_list = ['rho','u','v','w','p','T','rhou','rhov','rhow','rhoe', \
                'rho**2','uu','vv','ww','uv','uw','vw','vT','p**2','T**2',\
                'mu','divloc','divloc2']
    stats1,stats2 = 1,2
    stats = stats1
    if (var not in var_list):
        stats = stats2
    dict_info = read_info(dir_data)
    n_bl   = dict_info['nbloc']
    Re_in  = dict_info['Reref']
    u_in   = dict_info['Uref']
    T_in   = dict_info['Tref']
    p_in   = dict_info['Pref']
    rho_in = dict_info['Roref']
    mu_in  = dict_info['Muref']
    c_in   = dict_info['cref']
    nz     = dict_info['nz_bl1']
    L_ref  = dict_info['L_ref']
    # LS59
    # L_ref  = 0.08359031712
    # L_ref  = 0.02458326347
    # pitch  = 0.8495
    # Baumgartner
    L_ref  = 0.0150000 # proj chord
    # L_ref  = 0.0219935 # true chord
    # pitch  = 0.9162978573
    pitch  = 1.047197551
    norm   = u_in
    # isentropic Mach computation
    if dir_data.split('_')[-2]=='air':
        # air
        p0_in  = 1200000
        gam_in = 1.4
    else:
        # r134a
        if dir_data.split('_')[-1]=='newPR/':
            p0_in = 483000
        else:
            p0_in  = 560000
        gam_in = 1.08

    # if var[0]=='p':
    #     normvar = p_in
    if var[0]=='T' :
        normvar = T_in
    elif len(var)>=3 and var[:3]=='rho':
        normvar = rho_in
    elif var=='prr' or var=='g':
        normvar = 1
    elif var=='c':
        normvar = c_in
    else:
        normvar = norm**2
    nw = sum([dict_info['ny_bl3'],dict_info['nx_bl4'], \
              dict_info['ny_bl6'],dict_info['ny_bl7']])
    nw_loc = 0
    L_z = np.pi*L_ref
    #dz  = L_z/(nz-1)
    Tw,ovrht = 0,0

    data,data2 = d(),d()
    vmin,vmax = [],[]
    vmink,vmaxk = [],[]
    vminP,vmaxP = [],[]
    vminPc,vmaxPc = [],[]
    vminPs,vmaxPs = [],[]
    p0        = []
    xw,yw     = np.zeros((nw)),np.zeros((nw))
    xm1,xp1   = np.zeros((nw)),np.zeros((nw))
    dxw       = np.zeros((nw))
    pw,cpw    = np.zeros((nw)),np.zeros((nw))
    theta,r   = np.zeros((nw)),np.zeros((nw))
    muw,la    = np.zeros((nw)),np.zeros((nw))
    rhow,y1   = np.zeros((nw))+1,np.zeros((nw))
    x1_,y1_   = np.zeros((nw)),np.zeros((nw))
    x2_,y2_   = np.zeros((nw)),np.zeros((nw))
    s         = np.zeros((nw))
    dTx,dTy   = np.zeros((nw)),np.zeros((nw))
    Twa,tauw  = np.zeros((nw)),np.zeros((nw))
    prms      = np.zeros((nw))
    duxw,duyw,duzw = np.zeros((nw)),np.zeros((nw)),np.zeros((nw))
    dvxw,dvyw,dvzw = np.zeros((nw)),np.zeros((nw)),np.zeros((nw))
    dwxw,dwyw,dwzw = np.zeros((nw)),np.zeros((nw)),np.zeros((nw))
    tau = np.zeros((nw,2,2))

    # For regular grid interpolation/streamlines
    u_flat   = []
    v_flat   = []
    x_flat   = []
    y_flat   = []
    var_flat = []
    rho_flat = []
    p_flat = []
    T_flat = []
    k_flat   = []
    P_flat   = []
    P_c_flat = []
    P_s_flat = []
    M_is_flat = []
    M_flat = []
    p_wake_flat = []

    # Block caracteristics
    # /!\ wall direction is clockwise
    blcar = {}
    for bl in range(n_bl):
        bl+=1
        blcar[f'block_{bl}'] = {}
        if bl==3 or bl==4 or bl==6 or bl==7:
            blcar[f'block_{bl}']['wall'] = True
        else:
            blcar[f'block_{bl}']['wall'] = False

        if bl==3 or bl==6 or bl==7:
            # Wall location
            blcar[f'block_{bl}']['imax'] = True
            blcar[f'block_{bl}']['jmin'] = False
        if bl==4:
            # Wall location
            blcar[f'block_{bl}']['jmin'] = True
            blcar[f'block_{bl}']['imax'] = False
    
    # Extract data
    for bl in range(n_bl):
        bl+=1

        # Retrieve data from readvars
        nx,ny,nz,x,y,z = read_grid(dir_data+'grid_bl{}.bin'.format(bl),is_curv)
        bl_data,data   = extr_dict(is_RANS,stats1,bl,nx,ny,dir_data,data)
        bl_data2,data2 = extr_dict(is_RANS,stats2,bl,nx,ny,dir_data,data2)

        # isentropic Mach number
        if var=='M_is':
            data[f'block_{bl}']['M_is'] = np.sqrt(2/(gam_in-1)*\
                  ((p0_in/bl_data['p'])**((gam_in-1)/gam_in)-1))
            M_is_flat.append(data[f'block_{bl}']['M_is'].flatten())

        if var=='p_wake':
            data[f'block_{bl}']['p_wake'] = bl_data['p']/p0_in
            p_wake_flat.append(data[f'block_{bl}']['p_wake'].flatten())

        elif stats==1:
            data[f'block_{bl}'][var] = bl_data[var]
        else:
            data[f'block_{bl}'][var] = bl_data2[var]

        if var=='M':
            M_flat.append(data[f'block_{bl}']['M'].flatten())

        data[f'block_{bl}']['x'],data[f'block_{bl}']['y'] = x/L_ref,y/L_ref

        # Operations for rms
        # if len(var)>1 and var[-3:]=='**2':
        #     data[f'block_{bl}'][var] = np.sqrt(data[f'block_{bl}'][var]-bl_data[var[:-3]]**2)

        # Flattened coords
        data[f'block_{bl}']['x_flat'] = data[f'block_{bl}']['x'].flatten()
        data[f'block_{bl}']['y_flat'] = data[f'block_{bl}']['y'].flatten()
        x_flat.append(data[f'block_{bl}']['x'].flatten())
        y_flat.append(data[f'block_{bl}']['y'].flatten())

        # Extract wall data
        if blcar[f'block_{bl}']['wall']:
            if blcar[f'block_{bl}']['imax']:
                # Wall coordinates
                xw[nw_loc:nw_loc+ny] = x[-1,:]/L_ref
                yw[nw_loc:nw_loc+ny] = y[-1,:]/L_ref
                # Compute first cell height
                x1_[nw_loc:nw_loc+ny] = x[-2,:]/L_ref
                y1_[nw_loc:nw_loc+ny] = y[-2,:]/L_ref
                if bl==6:
                    yw[nw_loc:nw_loc+ny] +=-pitch
                    y1_[nw_loc:nw_loc+ny]+=-pitch
                y1[nw_loc:nw_loc+ny]  = (np.sqrt((x1_-xw)**2+(y1_-yw)**2))[nw_loc:nw_loc+ny]
                # Cp
                pw[nw_loc:nw_loc+ny] = bl_data['p'][-1,:]
                p0.append(pw.max())
                # Tw
                Twa[nw_loc:nw_loc+ny] = bl_data['T'][-1,:]
                # Cf
                if Cf or yp or Prms or BL:
                    muw[nw_loc:nw_loc+ny]  = bl_data['mu'][-1,:]
                    rhow[nw_loc:nw_loc+ny] = bl_data['rho'][-1,:]
                    duxw[nw_loc:nw_loc+ny] = bl_data2['rho*dux'][-1,:]
                    duyw[nw_loc:nw_loc+ny] = bl_data2['rho*duy'][-1,:]
                    duzw[nw_loc:nw_loc+ny] = bl_data2['rho*duz'][-1,:]
                    dvxw[nw_loc:nw_loc+ny] = bl_data2['rho*dvx'][-1,:]
                    dvyw[nw_loc:nw_loc+ny] = bl_data2['rho*dvy'][-1,:]
                    dvzw[nw_loc:nw_loc+ny] = bl_data2['rho*dvz'][-1,:]
                    dwxw[nw_loc:nw_loc+ny] = bl_data2['rho*dwx'][-1,:]
                    dwyw[nw_loc:nw_loc+ny] = bl_data2['rho*dwy'][-1,:]
                    dwzw[nw_loc:nw_loc+ny] = bl_data2['rho*dwz'][-1,:]
                    # In cartesian coordinates
                    #tau[nw_loc:nw_loc+ny,0,0] = bl_data2['Frhou'][-1,:]
                    #tau[nw_loc:nw_loc+ny,0,1] = bl_data2['Frhov'][-1,:]
                    #tau[nw_loc:nw_loc+ny,1,0] = bl_data2['Frhov'][-1,:]
                    #tau[nw_loc:nw_loc+ny,1,1] = bl_data2['Grhov'][-1,:]
                    tau[nw_loc:nw_loc+ny,0,0] = (muw*2*duxw/rhow)[nw_loc:nw_loc+ny]
                    tau[nw_loc:nw_loc+ny,0,1] = (muw*(duyw+dvxw)/rhow)[nw_loc:nw_loc+ny]
                    tau[nw_loc:nw_loc+ny,1,0] = (muw*(duyw+dvxw)/rhow)[nw_loc:nw_loc+ny]
                    tau[nw_loc:nw_loc+ny,1,1] = (muw*2*dvyw/rhow)[nw_loc:nw_loc+ny]

                    # P_rms wall
                    prms[nw_loc:nw_loc+ny] = np.sqrt((bl_data['p**2']-bl_data['p']**2))[-1,:]

                # dTdn
                if Nu:
                    dTx[nw_loc:nw_loc+ny] = bl_data2['dTx'][-1,:]
                    dTy[nw_loc:nw_loc+ny] = bl_data2['dTy'][-1,:]
                nw_loc+=ny

            if blcar[f'block_{bl}']['jmin']:
                # Wall coordinates
                xw[nw_loc:nw_loc+nx]    = x[:,0]/L_ref
                yw[nw_loc:nw_loc+nx]    = y[:,0]/L_ref
                # Compute first cell height
                x1_[nw_loc:nw_loc+nx]   = x[:,1]/L_ref
                y1_[nw_loc:nw_loc+nx]   = y[:,1]/L_ref
                y1[nw_loc:nw_loc+nx]    = (np.sqrt((x1_-xw)**2+(y1_-yw)**2))[nw_loc:nw_loc+nx]
                # Cp
                pw[nw_loc:nw_loc+nx] = bl_data['p'][:,0]
                p0.append(pw.max())
                # Tw
                Twa[nw_loc:nw_loc+nx] = bl_data['T'][:,0]
                # Cf
                if Cf or yp or Prms or BL:
                    muw[nw_loc:nw_loc+nx]  = bl_data['mu'][:,0]
                    rhow[nw_loc:nw_loc+nx] = bl_data['rho'][:,0]
                    duxw[nw_loc:nw_loc+nx] = bl_data2['rho*dux'][:,0]
                    duyw[nw_loc:nw_loc+nx] = bl_data2['rho*duy'][:,0]
                    duzw[nw_loc:nw_loc+nx] = bl_data2['rho*duz'][:,0]
                    dvxw[nw_loc:nw_loc+nx] = bl_data2['rho*dvx'][:,0]
                    dvyw[nw_loc:nw_loc+nx] = bl_data2['rho*dvy'][:,0]
                    dvzw[nw_loc:nw_loc+nx] = bl_data2['rho*dvz'][:,0]
                    dwxw[nw_loc:nw_loc+nx] = bl_data2['rho*dwx'][:,0]
                    dwyw[nw_loc:nw_loc+nx] = bl_data2['rho*dwy'][:,0]
                    dwzw[nw_loc:nw_loc+nx] = bl_data2['rho*dwz'][:,0]
                    # In cartesian coordinates
                    #tau[nw_loc:nw_loc+nx,0,0] = bl_data2['Frhou'][:,0]
                    #tau[nw_loc:nw_loc+nx,0,1] = bl_data2['Frhov'][:,0]
                    #tau[nw_loc:nw_loc+nx,1,0] = bl_data2['Frhov'][:,0]
                    #tau[nw_loc:nw_loc+nx,1,1] = bl_data2['Grhov'][:,0]
                    tau[nw_loc:nw_loc+nx,0,0] = (muw*2*duxw/rhow)[nw_loc:nw_loc+nx]
                    tau[nw_loc:nw_loc+nx,0,1] = (muw*(duyw+dvxw)/rhow)[nw_loc:nw_loc+nx]
                    tau[nw_loc:nw_loc+nx,1,0] = (muw*(duyw+dvxw)/rhow)[nw_loc:nw_loc+nx]
                    tau[nw_loc:nw_loc+nx,1,1] = (muw*2*dvyw/rhow)[nw_loc:nw_loc+nx]

                    # P_rms wall
                    prms[nw_loc:nw_loc+nx] = np.sqrt((bl_data['p**2']-bl_data['p']**2))[:,0]

                # dTdn
                if Nu:
                    dTx[nw_loc:nw_loc+nx] = bl_data2['dTx'][:,0]
                    dTy[nw_loc:nw_loc+nx] = bl_data2['dTy'][:,0]
                nw_loc+=nx

            # import matplotlib.pyplot as plt
            # plt.plot(xw,yw)
            # plt.show()

        # Operations for Reynolds stress
        if len(var)>1 and (var[0]==var[1] or var=='uv' or var=='uw' or var=='vw'):
            data[f'block_{bl}']['u'] = bl_data['u']
            data[f'block_{bl}']['v'] = bl_data['v']
            # data[f'block_{bl}']['w'] = bl_data['w']
            data[f'block_{bl}'][var] = data[f'block_{bl}'][var]-bl_data[var[0]]*bl_data[var[1]]
            norm = u_in**2
            # RANS computation of Reynolds stress
            #if is_RANS:
            #    data[f'block_{bl}']['mut'] = bl_data['mut']
            #    # /!\ uv = rho*<u'v'> /!\
            #    if var=='uv':
            #        data[f'block_{bl}'][var] = bl_data['mut']*(bl_data['duy']+bl_data['dvx'])
            #    elif var=='uu':
            #        data[f'block_{bl}'][var] = 2*bl_data['mut']*bl_data['dux']
            #    elif var=='vv':
            #        data[f'block_{bl}'][var] = 2*bl_data['mut']*bl_data['dvy']
            #    # Turbulent production and dissipation terms
            #    P = bl_data['mut']*(2*bl_data['dux']*bl_data['dux'] + \
            #                         (bl_data['duy']+bl_data['dvx'])*bl_data['duy'] + \
            #                         (bl_data['duy']+bl_data['dvx'])*bl_data['dvx'] + \
            #                        2*bl_data['dvy']*bl_data['dvy'])
            #    eps  = bl_data['mu']*(bl_data['dux']**2+bl_data['duy']**2+\
            #                          bl_data['dvx']**2+bl_data['dvy']**2)
            vmin_ = data[f'block_{bl}'][var].min()
            vmax_ = data[f'block_{bl}'][var].max()

        # Turbulent kinetic energy
        if k or sl:
            # TKE
            bl_data['util'] = bl_data['rhou']/bl_data['rho']
            bl_data['vtil'] = bl_data['rhov']/bl_data['rho']
            bl_data['wtil'] = bl_data['rhow']/bl_data['rho']
            data[f'block_{bl}']['k'] = 1/2*(bl_data['uu']+bl_data['util']**2- \
                                          2*bl_data['u'] *bl_data['util']   + \
                                            bl_data['vv']+bl_data['vtil']**2- \
                                          2*bl_data['v'] *bl_data['vtil']   + \
                                            bl_data['ww']+bl_data['wtil']**2- \
                                          2*bl_data['w'] *bl_data['wtil']     )
            data[f'block_{bl}']['k'] = data[f'block_{bl}']['k']/u_in**2
            # Production of TKE
            list1_ = ['u','v','w']
            list2_ = ['x','y','z']
            for vel in list1_:
                for coor_ in list2_:
                    bl_data[f'd{vel}til{coor_}'] = bl_data2[f'rho*d{vel}{coor_}']/\
                                                    bl_data['rho']
            P11 = (bl_data2[f'rho*uu']-bl_data['rhou']*bl_data['util'])*bl_data['dutilx']
            P12 = (bl_data2[f'rho*uv']-bl_data['rhov']*bl_data['util'])*bl_data['dutily']
            P13 = (bl_data2[f'rho*uw']-bl_data['rhow']*bl_data['util'])*bl_data['dutilz']
            P21 = (bl_data2[f'rho*uv']-bl_data['rhou']*bl_data['vtil'])*bl_data['dvtilx']
            P22 = (bl_data2[f'rho*vv']-bl_data['rhov']*bl_data['vtil'])*bl_data['dvtily']
            P23 = (bl_data2[f'rho*vw']-bl_data['rhow']*bl_data['vtil'])*bl_data['dvtilz']
            P31 = (bl_data2[f'rho*uw']-bl_data['rhou']*bl_data['wtil'])*bl_data['dwtilx']
            P32 = (bl_data2[f'rho*vw']-bl_data['rhov']*bl_data['wtil'])*bl_data['dwtily']
            P33 = (bl_data2[f'rho*ww']-bl_data['rhow']*bl_data['wtil'])*bl_data['dwtilz']
            data[f'block_{bl}']['P_k'] = -(P11+P12+P13+P21+P22+P23+P31+P32+P33)
            data[f'block_{bl}']['P_k'] = data[f'block_{bl}']['P_k']/rho_in/u_in**3*L_ref
            # Separate terms
            data[f'block_{bl}']['P_k_compr'] = -(P11+P22+P33)/rho_in/u_in**3*L_ref
            data[f'block_{bl}']['P_k_shear'] = data[f'block_{bl}']['P_k']-\
                                               data[f'block_{bl}']['P_k_compr']
            # Store
            vmink.append(data[f'block_{bl}']['k'].min())
            vmaxk.append(data[f'block_{bl}']['k'].max())
            vminP.append(data[f'block_{bl}']['P_k'].min())
            vmaxP.append(data[f'block_{bl}']['P_k'].max())
            data[f'block_{bl}']['k_flat'] = data[f'block_{bl}']['k'].flatten()
            data[f'block_{bl}']['P_k_flat'] = data[f'block_{bl}']['P_k'].flatten()
            k_flat.append(data[f'block_{bl}']['k'].flatten())
            P_flat.append(data[f'block_{bl}']['P_k'].flatten())
            # Separate terms
            vminPc.append(data[f'block_{bl}']['P_k_compr'].min())
            vmaxPc.append(data[f'block_{bl}']['P_k_compr'].max())
            vminPs.append(data[f'block_{bl}']['P_k_shear'].min())
            vmaxPs.append(data[f'block_{bl}']['P_k_shear'].max())
            data[f'block_{bl}']['P_c_flat'] = data[f'block_{bl}']['P_k_compr'].flatten()
            data[f'block_{bl}']['P_s_flat'] = data[f'block_{bl}']['P_k_shear'].flatten()
            P_c_flat.append(data[f'block_{bl}']['P_c_flat'])
            P_s_flat.append(data[f'block_{bl}']['P_s_flat'])
            # Others
            data[f'block_{bl}']['dutilx_flat'] = bl_data['dutilx'].flatten()

        # if var[0]=='T':
        #     Tw = Twa[0]
        #     ovrht = Tw/T_in
        #     data[f'block_{bl}'][var] = T_in-(data[f'block_{bl}'][var]-Tw)/(1-ovrht)

        # Operations for grid interpolation and streamlines
        if stream or yz or xz or BL or BL_scal:
            u_flat.append(bl_data['u'].flatten())
            v_flat.append(bl_data['v'].flatten())
            rho_flat.append(bl_data['rho'].flatten())
            p_flat.append(bl_data['p'].flatten())
            T_flat.append(bl_data['T'].flatten())
            var_flat.append(data[f'block_{bl}'][var].flatten())
            # Per block
            data[f'block_{bl}']['u_flat'] = bl_data['u'].flatten()
            data[f'block_{bl}']['v_flat'] = bl_data['v'].flatten()
            data[f'block_{bl}']['rho_flat'] = bl_data['rho'].flatten()
            data[f'block_{bl}']['p_flat'] = bl_data['p'].flatten()
            data[f'block_{bl}']['T_flat'] = bl_data['T'].flatten()
            data[f'block_{bl}']['var_flat'] = data[f'block_{bl}'][var].flatten()

        # Operations for Mach number and speed of sound
        data[f'block_{bl}']['u'] = bl_data['u']
        data[f'block_{bl}']['v'] = bl_data['v']
        data[f'block_{bl}']['rho'] = bl_data['rho']
        data[f'block_{bl}']['mu'] = bl_data['mu']
        if var!='M':
            data[f'block_{bl}']['M'] = bl_data2['M']
        data[f'block_{bl}']['vort']  =-bl_data2['b3']*L_ref/u_in

        vmin_ = data[f'block_{bl}'][var].min()
        vmax_ = data[f'block_{bl}'][var].max()
        vmin.append(vmin_)
        vmax.append(vmax_)

    # Add more useful stuff
    #######################

    # Inlet data
    data['n_bl'] = n_bl
    data['u_in'] = u_in
    data['p_in'] = p_in
    data['T_in'] = T_in
    data['mu_in'] = mu_in
    data['rho_in'] = rho_in
    data['vmin'] = min(vmin)
    data['vmax'] = max(vmax)
    data['L_ref'] = L_ref
    data['norm'] = norm
    data['normvar'] = normvar
    data['Tw'] = Tw
    data['ovrht'] = ovrht

    # Outlet pitchwise average
    i = 30
    rho_out = 0.5*(np.mean(data['block_8']['rho'][i,:])+np.mean(data['block_9']['rho'][i,:]))
    u_out = 0.5*(np.mean(data['block_8']['u'][i,:])+np.mean(data['block_9']['u'][i,:]))
    v_out = 0.5*(np.mean(data['block_8']['v'][i,:])+np.mean(data['block_9']['v'][i,:]))
    U_out = np.sqrt(u_out**2+v_out**2)
    mu_out = 0.5*(np.mean(data['block_8']['mu'][i,:])+np.mean(data['block_9']['mu'][i,:]))
    M_out = 0.5*(np.mean(data['block_8']['M'][i,:])+np.mean(data['block_9']['M'][i,:]))
    thet_out = np.arctan(v_out/u_out)*180/np.pi
    #L_ref = 23.239518e-3 # axial chord
    Re_out = rho_out*U_out*L_ref/mu_out
    print('Re_out =',Re_out)
    print('M_out =',M_out)
    print('thet_out =',thet_out)

    # Wall coordinates
    def rearrange(x,d):
        len_ = d['ny_bl3']+d['nx_bl4']
        # save bl6 coords
        x_ = np.zeros(d['ny_bl6'])
        x_ = x[len_:len_+d['ny_bl6']].copy()
        # rotate array
        x[len_:len_+d['ny_bl7']] = x[-d['ny_bl7']:].copy()
        x[-d['ny_bl6']:] = x_.copy()

        return x
    xw = rearrange(xw,dict_info)
    yw = rearrange(yw,dict_info)
    pw = rearrange(pw,dict_info)

    # Translate everything so stagnation point is @ (0,0)
    ind_p0 = np.argmax(pw) # index of P0 to start wall coords
    xw_p0 = xw[ind_p0]
    yw_p0 = yw[ind_p0]
    s = np.hstack((np.array([0]),np.sqrt((xw[1:]-xw[:-1])**2+(xw[1:]-xw[:-1])**2)[ind_p0:],\
                   np.sqrt((xw[1:]-xw[:-1])**2+(xw[1:]-xw[:-1])**2)[:ind_p0]))
    s = np.cumsum(s)
    # Rotate coordinates by stagger angle
    thet = 43.3*np.pi/180
    coords = np.vstack((xw,yw))
    R = np.array([[np.cos(thet),-np.sin(thet)],[np.sin(thet),np.cos(thet)]])
    coords = np.matmul(R,coords)
    xw = coords[0,:]
    yw = coords[1,:]
    xw = xw-xw.min()
    xw = xw/(xw.max()-xw.min())
    yw = yw/(xw.max()-xw.min())
    # Store
    data['xw'] = xw
    data['yw'] = yw
    data['s']  = s
    nwall = np.zeros((nw,2))
    with open(dir_data+'norm_surf.dat','r') as f:
        dat = np.loadtxt(f)
        nxwalldl = dat[:,0]
        nywalldl = dat[:,1]
        nxwall = dat[:,2]
        nywall = dat[:,3]
        dl     = dat[:,4]
        ijacob = dat[:,5]
        nwall[:,0],nwall[:,1] = nxwall,nywall
        nwalldl = np.sqrt(nxwalldl**2+nywalldl**2)
        data['nwall'] = nwall

    # Wall data
    data['p0'] = p0_in
    data['p/p0'] = pw/data['p0']
    # Wall friction
    for i in range(nw):
       tauw[i] = (tau[i]@nwall[i])[0]#*ijacob[i]*nwalldl[i]
    tauw = rearrange(tauw,dict_info)
    y1 = rearrange(y1,dict_info)
    x1 = rearrange(nwalldl,dict_info)/L_ref
    rhow = rearrange(rhow,dict_info)
    muw = rearrange(muw,dict_info)
    utau = np.sqrt(abs(tauw)/rhow)
    data['cf'] = -tauw/(0.5*rho_in*u_in**2)

    # Non dimensional wall data
    normw = utau
    data['yp'] = y1*L_ref*utau/muw*rhow
    data['xp'] = x1*L_ref*utau/muw*rhow
    data['tauw'] = tauw
    data['utau'] = utau
    data['rhow'] = rhow
    data['muw']  = muw

    # Nusselt number
    if Nu:
        data['Nus']  = sum((dTx*nxwalldl+dTy*nywalldl))/(Tw-T_in)/np.pi
        data['Nusw'] = (dTx*nxwall+dTy*nywall)*L_ref/(Tw-T_in)

    # Flattened data
    data['x_flat'] = np.hstack(x_flat)
    data['y_flat'] = np.hstack(y_flat)
    if var=='M_is':
        data['M_is_flat'] = np.hstack(M_is_flat)
        np.nan_to_num(data['M_is_flat'],0)
    if var=='M':
        data['M_flat'] = np.hstack(M_flat)
    if var=='p_wake':
        data['p_wake_flat'] = np.hstack(p_wake_flat)

    if stream or yz or xz or BL or BL_scal:
        data['u_flat'] = np.hstack(u_flat)
        data['v_flat'] = np.hstack(v_flat)
        data['rho_flat'] = np.hstack(rho_flat)
        data['p_flat'] = np.hstack(p_flat)
        data['T_flat'] = np.hstack(T_flat)
        data['var_flat'] = np.hstack(var_flat)

    if k or sl:
        data['vmink'] = min(vmink)
        data['vmaxk'] = max(vmaxk)
        data['vminP'] = min(vminP)
        data['vmaxP'] = max(vmaxP)
        data['vminPc'] = min(vminPc)
        data['vmaxPc'] = max(vmaxPc)
        data['vminPs'] = min(vminPs)
        data['vmaxPs'] = max(vmaxPs)
        data['k_flat'] = np.hstack(k_flat)
        data['P_flat'] = np.hstack(P_flat)
        data['P_c_flat'] = np.hstack(P_c_flat)
        data['P_s_flat'] = np.hstack(P_s_flat)

    return data

def turb_total_q(dir_data,is_curv,is_RANS,var,model):

    dict_info = read_info(dir_data)
    n_bl   = dict_info['nbloc']
    L_ref  = dict_info['L_ref']
    # LS59
    # L_ref  = 0.08359031712
    # L_ref  = 0.02458326347
    # pitch  = 0.8495
    # Baumgartner
    L_ref  = 0.0150000 # proj chord
    # L_ref  = 0.0219935 # true chord
    # pitch  = 0.9162978573
    pitch  = 1.047197551

    # Data storage
    data,data2 = d(),d()

    # For regular grid interpolation/streamlines
    u_flat = []
    v_flat = []
    x_flat = []
    y_flat = []
    vmin   = []
    vmax   = []

    # Extract data
    for bl in range(n_bl):
        bl+=1

        # Retrieve data from readvars
        nx,ny,nz,x,y,z = read_grid(dir_data+'grid_bl{}.bin'.format(bl),is_curv)
        bl_data,data   = extr_dict(is_RANS,1,bl,nx,ny,dir_data,data)
        bl_data2,data2 = extr_dict(is_RANS,2,bl,nx,ny,dir_data,data2)
        data[f'block_{bl}']['x'],data[f'block_{bl}']['y'] = x/L_ref,y/L_ref

        # Flattened coords
        data[f'block_{bl}']['x_flat'] = data[f'block_{bl}']['x'].flatten()
        data[f'block_{bl}']['y_flat'] = data[f'block_{bl}']['y'].flatten()
        x_flat.append(data[f'block_{bl}']['x'].flatten())
        y_flat.append(data[f'block_{bl}']['y'].flatten())

        # Operations for grid interpolation and streamlines
        if yz or xz:
            u_flat.append(bl_data['u'].flatten())
            v_flat.append(bl_data['v'].flatten())
            # Per block
            data[f'block_{bl}']['u_flat'] = bl_data['u'].flatten()
            data[f'block_{bl}']['v_flat'] = bl_data['v'].flatten()

        # Total quantities
        data[f'block_{bl}']['p0'] = np.zeros((nx,ny))
        data[f'block_{bl}']['T0'] = np.zeros((nx,ny))
        data[f'block_{bl}']['ro0'] = np.zeros((nx,ny))

        ### PFG ###
        if model=='PFG':
            data[f'block_{bl}']['p0'],\
            data[f'block_{bl}']['T0'],\
            data[f'block_{bl}']['ro0'],\
            = total_pfg.total(bl_data['p'],bl_data['T'],\
                                  bl_data['rho'],bl_data2['M'])

            # Post normal shock quantities
            data[f'block_{bl}']['p2'], \
            data[f'block_{bl}']['T2'], \
            data[f'block_{bl}']['ro2'],\
            data[f'block_{bl}']['e2'], \
            data[f'block_{bl}']['s2'], \
            data[f'block_{bl}']['h2'], \
            data[f'block_{bl}']['M2'], \
            = post_shock_pfg.post_shock(bl_data['p'],bl_data['T'],\
                               bl_data['rho'],bl_data['u']**2+\
                               bl_data['v']**2+bl_data['w']**2,\
                               bl_data2['e'],bl_data2['M'])

            # Post normal shock total quantities
            data[f'block_{bl}']['p02'],\
            data[f'block_{bl}']['T02'],\
            data[f'block_{bl}']['ro02'],\
            = total_pfg.total(data[f'block_{bl}']['p2'],\
                               data[f'block_{bl}']['T2'],\
                               data[f'block_{bl}']['ro2'],\
                               data[f'block_{bl}']['M2'])

            vmin_ = data[f'block_{bl}'][var].min()
            vmax_ = data[f'block_{bl}'][var].max()
            vmin.append(vmin_)
            vmax.append(vmax_)

        ### PRS ###
        else:
            for i in range(nx):
                for j in range(ny):
                    data[f'block_{bl}']['p0'][i,j],\
                    data[f'block_{bl}']['T0'][i,j],\
                    data[f'block_{bl}']['ro0'][i,j],\
                    = total_prsv.total(bl_data['p'][i,j],bl_data['T'][i,j],\
                                       bl_data['rho'][i,j],bl_data['u'][i,j]**2+\
                                       bl_data['v'][i,j]**2+bl_data['w'][i,j]**2,\
                                       bl_data2['e'][i,j],bl_data2['s'][i,j],\
                                       bl_data2['M'][i,j])

            # Post normal shock quantities
            data[f'block_{bl}']['p2'] = np.zeros((nx,ny))
            data[f'block_{bl}']['T2'] = np.zeros((nx,ny))
            data[f'block_{bl}']['ro2'] = np.zeros((nx,ny))
            data[f'block_{bl}']['e2'] = np.zeros((nx,ny))
            data[f'block_{bl}']['s2'] = np.zeros((nx,ny))
            data[f'block_{bl}']['h2'] = np.zeros((nx,ny))
            data[f'block_{bl}']['V2'] = np.zeros((nx,ny))
            data[f'block_{bl}']['M2'] = np.zeros((nx,ny))
            for i in range(nx):
                for j in range(ny):
                    if bl_data2['M'][i,j]>1:
                        data[f'block_{bl}']['p2'][i,j],\
                        data[f'block_{bl}']['T2'][i,j],\
                        data[f'block_{bl}']['ro2'][i,j],\
                        data[f'block_{bl}']['V2'][i,j],\
                        data[f'block_{bl}']['e2'][i,j],\
                        data[f'block_{bl}']['s2'][i,j],\
                        data[f'block_{bl}']['h2'][i,j],\
                        data[f'block_{bl}']['M2'][i,j],\
                        = post_shock_prsv.post_shock(bl_data['p'][i,j],bl_data['T'][i,j],\
                                           bl_data['rho'][i,j],bl_data['u'][i,j]**2+\
                                           bl_data['v'][i,j]**2+bl_data['w'][i,j]**2,\
                                           bl_data2['e'][i,j],bl_data2['M'][i,j])

            # Post normal shock total quantities
            data[f'block_{bl}']['p02'] = np.zeros((nx,ny))
            data[f'block_{bl}']['T02'] = np.zeros((nx,ny))
            data[f'block_{bl}']['ro02'] = np.zeros((nx,ny))
            for i in range(nx):
                for j in range(ny):
                    if data[f'block_{bl}']['M2'][i,j]>0:
                        data[f'block_{bl}']['p02'][i,j],\
                        data[f'block_{bl}']['T02'][i,j],\
                        data[f'block_{bl}']['ro02'][i,j],\
                        = total_prsv.total(data[f'block_{bl}']['p2'][i,j],\
                                           data[f'block_{bl}']['T2'][i,j],\
                                           data[f'block_{bl}']['ro2'][i,j],\
                                           data[f'block_{bl}']['V2'][i,j],\
                                           data[f'block_{bl}']['e2'][i,j],\
                                           data[f'block_{bl}']['s2'][i,j],\
                                           data[f'block_{bl}']['M2'][i,j])

            vmin_ = data[f'block_{bl}'][var].min()
            vmax_ = data[f'block_{bl}'][var].max()
            vmin.append(vmin_)
            vmax.append(vmax_)

    data['n_bl'] = n_bl
    data['vmin'] = min(vmin)
    data['vmax'] = max(vmax)
    
    return data

# Extract data from stats (degas)
#########################

def read_stats_udegas(dir_data,var,field,stream,xz,yz,Lx,Cp,Cf):
    print("Reading stats...")
    file_input = "tecSTATS.plt"
    nx = np.array([37, 153, 37, 161, 113])-1
    ny = np.array([41, 113, 41, 49, 73])-1
    le = 19

    # Critical values for PRS
    rhoc = 606.804
    Pc = 1.869e6

    # Nb of blocks
    n_bl = len(nx)

    var_list = ['x','y','rho','M','sig','cp','p','g','a','u','v','uu','vv','uv',\
                'cf','qw','mut','vvpr','uvpr','truc']
    stats = d()
    stat  = d() # individual stat field
    p_b   = 0 # previous block
    vmin,vmax = [],[]
        
    # Extract and store data
    with open(dir_data+file_input, 'r') as f:
        for j in range(6):
            f.readline()
        text = f.readlines()
        for i in range(n_bl):
            if i==0:
                data = np.loadtxt(text[:nx[i]*ny[i]])
            else:
                p_b += nx[i-1]*ny[i-1]+1
                data = np.loadtxt(text[p_b:p_b+nx[i]*ny[i]])
            stat['block_{}'.format(str(i+1))] = data

    # Attribute data to each variable
    i=0
    for bl,data in stat.items():
        stats[bl] = d()
        for k,var_ in enumerate(var_list):
            if bl=='block_1' or bl=='block_3':
                stats[bl][var_] = data[:,k].reshape((nx[i],ny[i]))
            else:
                stats[bl][var_] = data[:,k].reshape((ny[i],nx[i]))
            if var_=='x' or var_=='y':
                stats['block_{}'.format(bl.split('_')[-1])][var_] = stats[bl][var_]
            if var_==var:
                vmin.append(stats[bl][var].min())
                vmax.append(stats[bl][var].max())
        i+=1

    # Add more useful stuff
    #######################
   
    stats['n_bl'] = n_bl
    stats['u_in'] = stats['block_5']['u'][0,int(ny[-1]/2)]
    stats['rho_in'] = stats['block_5']['rho'][0,int(ny[-1]/2)]
    stats['M_in'] = stats['block_5']['M'][0,int(ny[-1]/2)]
    stats['p_in'] = stats['block_5']['p'][0,int(ny[-1]/2)]
    stats['p0'] = stats['block_4']['p'].max()
    stats['vmin'] = min(vmin)
    stats['vmax'] = max(vmax)
    stats['norm'] = 1 # For compatibility
    stats['normvar'] = 1
    stats['nwall'] = 1

    # Re-compute cp
    for bl in range(n_bl):
        bl = f'block_{bl+1}'
        # stats[bl]['cp'] = (stats[bl]['p']-stats['p_in'])/(stats['p0']-stats['p_in'])
        stats[bl]['cp'] = (stats[bl]['p']-stats['p_in'])/(0.5*stats['rho_in']*stats['u_in']**2)

    # Wall data
    xw = np.concatenate((stats['block_4']['x'][0][le:],stats['block_4']['x'][0][:le]))
    stats['xw'] = xw[:int(xw.size/2.)+1]
    stats['yw'] = np.concatenate((stats['block_4']['y'][0][le:], \
                                  stats['block_4']['y'][0][:le]))[:stats['xw'].size]
    stats['cp'] = np.concatenate((stats['block_4']['cp'][0][le:], \
                                  stats['block_4']['cp'][0][:le]))[:stats['xw'].size][::-1]
    stats['cf'] = -np.concatenate((stats['block_4']['cf'][0][le:], \
                                   stats['block_4']['cf'][0][:le]))[:stats['xw'].size][::-1]
#    if dir_data.split('/')[6]=='PRS':
#        stats['cf'] = stats['cf']/(Pc)*1.3e-5
    stats['r']  = np.sqrt(stats['xw']**2+stats['yw']**2)
    stats['theta'] = 180.-np.arccos(stats['xw']/stats['r'])*180./np.pi
    stats['theta'] = stats['theta'][::-1]
    stats['theta_sep'] = 1
    stats['cd(cp)'] = si.trapezoid(np.sin(stats['theta']*np.pi/180),stats['cp'])
    print('Cd(cp) =',stats['cd(cp)'])

    return stats


# Fluid data
############

def fluid_data(fluid):
    if fluid=='PRS':
        Tc = 441.81
        Pc = 1.869e+06
        rhoc = 606.8040

        return Tc,Pc,rhoc


# Read vorticity criterias
##########################

def read_vort(dir_data,crit):
    print("Reading vorticity criterias...")
    dict_info = read_info(dir_data)
    n_bl = dict_info['nbloc']
    Re_in  = dict_info['Reref']
    rho_in = dict_info['Roref']
    u_in   = dict_info['Uref']
    mu_in  = dict_info['Muref']
    L_ref  = dict_info['L_ref']
    crit_list = ['Q','lam2']
    i_crit = crit_list.index(crit)

    # For vorticity interpolation n regular cartesian grid
    nw = sum([dict_info['nx_bl{}'.format(i+1)] for i in range(4)])
    nw_loc = 0
    vmin,vmax = [],[]
    xw,yw = np.zeros((nw)),np.zeros((nw))
   
    vort,vort_list = d(),[]
    crit_list,x_flat,y_flat = [],[],[]
    dtype = np.dtype('f8')
    for bl in range(n_bl):
        bl+=1
        f = open(dir_data+'vort_bl{}.bin'.format(bl),'rb')
        vort[f'block_{bl}'] = d()
        vort[f'block_{bl}'] = d()
        
        # Extract data from vort files and get grids
        nx,ny,nz,x,y,z = read_grid(dir_data+'grid_bl{}.bin'.format(bl),True)
        temp_vort = np.fromfile(f,dtype=dtype,count=nx*ny*nz)
        temp_crit = np.fromfile(f,dtype=dtype,count=(i_crit+1)*nx*ny*nz)\
                                [i_crit*nx*ny*nz:]
        # For paraview
        vort[f'block_{bl}']['vort'] = temp_vort.reshape((nx,ny,nz),order='F')
        vort[f'block_{bl}'][crit] = temp_crit.reshape((nx,ny,nz),order='F')
        vort[f'block_{bl}']['x'] = np.repeat(x[:,:,np.newaxis],nz,axis=2) 
        vort[f'block_{bl}']['y'] = np.repeat(y[:,:,np.newaxis],nz,axis=2) 
        vort[f'block_{bl}']['z'] = z
        vort[f'block_{bl}']['nx'],vort[f'block_{bl}']['ny'] = nx,ny
        # For python 3D plotting
        x_flat.append(x.T.flatten()/L_ref)
        y_flat.append(y.T.flatten()/L_ref)
        vort_list.append(temp_vort.reshape((nx*ny,nz),order='F').T)
        crit_list.append(temp_crit.reshape((nx*ny,nz),order='F').T)
        vmin.append(vort[f'block_{bl}'][crit].min())
        vmax.append(vort[f'block_{bl}'][crit].max())

        f.close()

        # For array mask
        if bl<=4:
            xw[nw_loc:nw_loc+nx] = x[:nx,0]/L_ref
            yw[nw_loc:nw_loc+nx] = y[:nx,0]/L_ref
            nw_loc+=nx

    # Add more useful stuff
    #######################

    # General information
    vort['n_bl'] = n_bl
    vort['u_in'] = u_in
    vort['L_ref'] = L_ref
    vort['vmin'] = min(vmin)
    vort['vmax'] = max(vmax)
    vort['nz'] = nz

    # Finalize flattening
    vort['vort'] = np.hstack(vort_list)
    vort[crit] = np.hstack(crit_list)
    vort['x']  = np.hstack(x_flat)
    vort['y']  = np.hstack(y_flat)

    # Wall data
    vort['xw'] = xw
    vort['yw'] = yw

    return vort


# Read sensors
##############

def read_sensors(dir_data,var,sens_nb,sens_bl,iter_start,iter_stop,n_planes,auto,every):
    print("Reading sensor output...")
    f = open(f'{dir_data}/line_{sens_nb}_sol_bl{sens_bl}.bin','r')
    dtype = np.dtype('f8')
    sensor = d()
    dict_info = read_info(dir_data)
    nz = dict_info['nz_bl1']
    var_list = ['u','v','w','p']
    nvars = len(var_list)

    # Collect data from sensor file
    data = np.fromfile(f,dtype=dtype,count=-1)
    for i_var,var_ in enumerate(var_list):
        sensor[var_] = d()
        for plane in range(nz):
            sensor[var_][f'plane_{plane+1}'] = data[plane+i_var*nz::nz*nvars]

    # Compute statistical properties
    stats1 = 1
    dict_info = read_info(dir_data)
    n_bl   = dict_info['nbloc']
    Re_in  = dict_info['Reref']
    u_in   = dict_info['Uref']
    T_in   = dict_info['Tref']
    p_in   = dict_info['Pref']
    rho_in = dict_info['Roref']
    mu_in  = dict_info['Muref']
    c_in   = dict_info['cref']
    L_ref  = dict_info['L_ref']
    nz     = dict_info['nz_bl1']
    dt     = dict_info['dt']
    L_z = L_ref
    dz  = L_z/(nz-1)
    geom = dir_data.split('/')[6][:-2]
    fluid = dir_data.split('/')[5]
    data,data2 = d(),d()
    vmin,vmax = [],[]

    # Retrieve data from readvars
    nx,ny,nz,x,y,z = read_grid(dir_data+f'grid_bl{sens_bl}.bin',True)
    bl_data,data = extr_dict(False,stats1,sens_bl,nx,ny,dir_data,data)
    vmin_ = bl_data[var].min()
    vmax_ = bl_data[var].max()
    var_avgs = [np.mean(np.array([np.mean(sensor[var][f'plane_{i+1}'][iter_start:iter_stop]) \
                    for i in range(n_planes)]),axis=0) for j,var in enumerate(var_list)]
    # avg = np.mean(np.array([np.mean(sensor[var][f'plane_{i+1}'][iter_start:iter_stop]) \
    #                 for i in range(n_planes)]),axis=0)
    # if var=='p':
    #     R = bl_data[var][ind_i,ind_j]-avg**2
    # else:
    #     R = bl_data[var+var][ind_i,ind_j]-avg**2 # Reynolds stress

    # Get time and iterations
    niter = np.linspace(0,sensor[var]['plane_1'].size,sensor[var]['plane_1'].size).astype(int)
    dtstar = dt*u_in/L_ref*every
    tstar = niter*dtstar
    # niter,tstar = read_coeffs(True,dir_data+'coeff.dat',1)
    # every_clcd = int(niter[1]-niter[0])
    # if every_clcd!=1:
    #     dtstar = (tstar[1]-tstar[0])/every_clcd
    #     niter = np.linspace(niter[0],niter[-1]-1,(niter.size-1)*every_clcd)
    #     tstar = np.linspace(tstar[0],tstar[-1]-dtstar,(tstar.size-1)*every_clcd)
    # else:
    #     dtstar = tstar[1]-tstar[0]

    # if sensors recorded every N iterations
    # every = niter.size//sensor[var]['plane_1'].size

    # Start after transient
    niter = niter[iter_start:iter_stop]
    tstar = tstar[iter_start:iter_stop]
    tau   = tstar-tstar.min()
    # iter_start = iter_start//every#_clcd
    # iter_stop  = iter_stop//every#_clcd

    # Data storage
    sensor[var]['cross_corr']  = d()
    sensor[var]['auto_cov']    = d()
    sensor[var]['auto_corr']   = d()
    sensor[var]['auto_corr_z'] = d()
    sensor[var]['E'] = d()
    sensor[var]['K'] = d()
    sensor[var]['T'] = d()

    # Autocorrelations
    # ================
    import matplotlib.pyplot as plt
    if auto:

        # Loop over planes
        for plane in range(n_planes):
            flu_tot = sensor[var][f'plane_{plane+1}'][iter_start:iter_stop]#-avg # Total fluctuating component

            # Energy spectrum
            # ===============
            N = niter.size
            E,k = plt.psd(sensor[var][f'plane_{plane+1}'][iter_start:iter_stop],NFFT=N,\
                             Fs=1/dtstar,scale_by_freq=True,return_line=False,detrend='mean')
            sensor[var]['E'][f'plane_{plane+1}'] = E
            sensor[var]['K'][f'plane_{plane+1}'] = k

        # # Loop over planes
        # for plane in range(n_planes):
        #     flu_tot = sensor[var][f'plane_{plane+1}'][iter_start:iter_stop]-avg # Total fluctuating component

        #     # Autocorrelation for a range of offset times
        #     # ===============
        #     # for ind_off in range(niter.size): # time offset indices
        #     #     tstar_off = tstar[ind_off] # non-dim time offset
        #     #     # Offset sample
        #     #     if ind_off==0:
        #     #         flu = flu_tot
        #     #     else:
        #     #         flu = flu_tot[:-ind_off]
        #     #     flu_off = flu_tot[ind_off:]
        #     #     # Compute function
        #     #     rms       = np.sqrt(np.mean(flu**2))
        #     #     rms_off   = np.sqrt(np.mean(flu_off**2))
        #     #     auto_cov.append(np.mean(flu*flu_off))
        #     #     auto_corr.append(auto_cov[ind_off]/(rms*rms_off))
        #     # Normalise result by the first autocorr for 0 offset
        #     auto_corr = np.correlate(flu_tot,flu_tot,mode='full')[niter.size-1:]
        #     auto_corr = auto_corr/auto_corr[0]

        #     # Energy spectrum
        #     # ===============
        #     N = niter.size
        #     E = 1./2./np.pi*np.fft.fft(auto_corr)
        #     E = np.abs(E)[:N//2]**2
        #     k = 2.*np.pi*np.fft.fftfreq(N,dtstar)[:N//2]
        #     sensor[var]['E'][f'plane_{plane+1}'] = E
        #     sensor[var]['K'][f'plane_{plane+1}'] = k

        #     # Time integral scale
        #     # ====
        #     T = si.trapezoid(auto_corr,dx=dtstar)
        #     sensor[var]['T'][f'plane_{plane+1}'] = T
        #     sensor[var]['auto_corr'][f'plane_{plane+1}'] = auto_corr

        # Autocorrelation for a range of offset spanwise locations
        # ===============
        for i,var in enumerate(var_list):
            flu_1 = sensor[var][f'plane_1'][iter_start:iter_stop]-var_avgs[i]
            auto_corr_z = []
            auto_corr_z2 = []
            for plane in range(nz):
                flu_off = sensor[var][f'plane_{plane+1}'][iter_start:iter_stop]-var_avgs[i]
                # Compute function
                auto_corr_z.append(np.corrcoef(flu_1,flu_off)[0,1])
                #auto_corr_z2.append(np.mean(flu_off*flu_1))

        # sig = np.zeros(nz)
        # fft = np.zeros((N,nz//2-1))
        # dz_ = 1
        # for i in range(N):
        #     for plane in range(nz):
        #         sig[plane] = sensor[var][f'plane_{plane+1}'][i]-avg

        #     # Compute spatial FFT
        #     spa_fft = np.fft.fft(sig)
        #     spa_fft = np.abs(spa_fft)[1:nz//2]#**2
        #     fft[i]  = spa_fft
        # auto_corr_z = np.mean(fft,axis=0)/np.mean(fft,axis=0).max()

            sensor[var]['auto_corr_z'] = auto_corr_z/max(auto_corr_z)
            #sensor[var]['auto_corr_z2'] = auto_corr_z2/max(auto_corr_z2)

    # Crosscorrelations
    # =================
    else:
    
        # Total fluctuating component
        flu_tot1 = sensor[var][f'plane_1'][iter_start:iter_stop]-avg
        flu_tot2 = sensor[var][f'plane_125'][iter_start:iter_stop]-avg
        plt.figure()
        plt.plot(flu_tot1)
        plt.plot(flu_tot2)
        plt.show()
        # Normalise result by the max crosscorr for 0 offset
        cross_corr = np.correlate(flu_tot1,flu_tot2,mode='full')[niter.size-1:]
        cross_corr = cross_corr/max(cross_corr)
        sensor[var]['cross_corr'][f'plane_1_2'] = cross_corr

    # Add useful stuff
    sensor['niter'] = niter
    sensor['dtstar'] = dtstar
    sensor['tstar'] = tstar
    sensor['every'] = every
    sensor['tau']   = tau

    return sensor


# Compute sensor PDF
####################

def read_sensors_PDF(dir_data,sens_nb,sens_bl,ind_i,ind_j,iter_start,iter_stop,n_planes):
    print("Reading sensor output...")
    f = open(f'{dir_data}/line_{sens_nb}_sol_bl{sens_bl}.bin','r')
    dtype = np.dtype('f8')
    sensor = d()
    dict_info = read_info(dir_data)
    nz = dict_info['nz_bl1']
    var_list = ['u','v','w','p']
    nvars = len(var_list)

    # Collect data from sensor file
    data = np.fromfile(f,dtype=dtype,count=-1)
    for i_var,var_ in enumerate(var_list):
        sensor[var_] = d()
        for plane in range(nz):
            sensor[var_][f'plane_{plane+1}'] = data[plane+i_var*nz::nz*nvars]

    # Compute statistical properties
    stats1 = 1
    dict_info = read_info(dir_data)
    n_bl   = dict_info['nbloc']
    Re_in  = dict_info['Reref']
    u_in   = dict_info['Uref']
    T_in   = dict_info['Tref']
    p_in   = dict_info['Pref']
    rho_in = dict_info['Roref']
    mu_in  = dict_info['Muref']
    c_in   = dict_info['cref']
    L_ref  = dict_info['L_ref']
    nz     = dict_info['nz_bl1']
    dt     = dict_info['dt']
    L_z = L_ref
    dz  = L_z/(nz-1)
    geom = dir_data.split('/')[6][:-2]
    fluid = dir_data.split('/')[5]
    if fluid=='PRS':
        Tc,Pc,rhoc = fluid_data(fluid)
    
    data,data2 = d(),d()
    vmin,vmax = [],[]

    # Retrieve data from readvars
    nx,ny,nz,x,y,z = read_grid(dir_data+f'grid_bl{sens_bl}.bin',True)
    bl_data,data = extr_dict(False,stats1,sens_bl,nx,ny,dir_data,data)
    avg_u = np.mean(np.array([np.mean(sensor['u'][f'plane_{i+1}'][iter_start:iter_stop]) \
                    for i in range(n_planes)]),axis=0)
    avg_v = np.mean(np.array([np.mean(sensor['v'][f'plane_{i+1}'][iter_start:iter_stop]) \
                    for i in range(n_planes)]),axis=0)

    # Angle between u' and w'   
    sensor['phi'] = d()
    for plane in range(nz):
        sensor['phi'][f'plane_{plane+1}'] = np.arctan((sensor['v'][f'plane_{plane+1}']-avg_w)/ \
                                                      (sensor['u'][f'plane_{plane+1}']-avg_u)  \
                                                      )*180/np.pi
    sensor_mean = np.mean(np.array([sensor['phi'][f'plane_{i+1}'] \
                  for i in range(n_planes)]),axis=0)
    kde = sa.nonparametric.KDEUnivariate(sensor_mean)
    kde.fit()

    return sensor_mean,kde.support,kde.density


def write_pvd(file_name,nb,bl,nb_chknpts,is_fluct):
    f = open(f"{file_name}.pvd", "w")
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="Collection" version="0.1"\n')
    f.write('         byte_order="LittleEndian"\n')
    f.write('         compressor="vtkZLibDataCompressor">\n')
    f.write('  <Collection>\n')
    for i in range(nb_chknpts):
        i+=1
        f.write(f'    <DataSet timestep="{i}" group="" part="0"\n')
        if is_fluct:
            f.write(f'             file="par_volume_{str(nb).rjust(3,"0")}_{i}_bl{bl}_fluct.vts"/>\n')
        else:
            f.write(f'             file="par_volume_{str(nb).rjust(3,"0")}_{i}_bl{bl}.vts"/>\n')
    f.write('  </Collection>\n')
    f.write('</VTKFile>')
    f.close()
