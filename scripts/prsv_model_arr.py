import numpy as np

# PRSV model
#===========

fluid = 'air'
# fluid = 'novec649'

with open(f'feos_{fluid}.ini','r') as f1:
	lines = f1.readlines()
	start = -14
	end = -1
	Tc = float(lines[1][start:end])
	pc = float(lines[2][start:end])
	roc= float(lines[3][start:end])
	z  = float(lines[4][start:end])
	m  = float(lines[5][start:end])
	di = float(lines[6][start:end])
	Tb = float(lines[7][start:end])
	om = float(lines[8][start:end])
	cvinf = float(lines[9][start:end])
	nexp  = float(lines[11][start:end])
	rg  = float(lines[12][start:end])
	Prr = float(lines[13][start:end])
	gam = float(lines[14][start:end])

# Constants
apr= 0.457235*(Tc*rg)**2/pc
bpr= 0.077796*rg*Tc/pc
L0 = 0.378893
L1 = 1.4897153
L2 = 0.17131848
L3 = 0.0196554
Kpr=L0+L1*om-L2*om**2+L3*om**3

def alp(T):
	return (1.0+Kpr*(1.0-np.sqrt(abs(T/Tc))))**2

def avcalc_tro(T,ro):
#===============================================================================
 #> Compute isobaric expansion coefficient from T and rho
 #> - PRS EOS -
#===============================================================================
	# specific volume
	v= 1.0/ro

	dpdv= -rg*T/(v-bpr)**2 \
	  + 2.0*apr*alp(T)*(v+bpr)/(v**2+2.0*bpr*v-bpr**2)**2

	dalpdT=2.0*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))*(-Kpr/(2.0*(np.sqrt(T*Tc))))

	dpdT= rg/(v-bpr)-(dalpdT)*apr/(v**2+2.0*bpr*v-bpr**2)

	return -ro*dpdT/dpdv

def c2calc_tro(T,ro):
#===============================================================================
#> Compute speed of sound from T and rho
#> - PRS EOS -
#===============================================================================
	# specific volume
	v= 1.0/ro

	dalpdT=2.0*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))*(-Kpr/(2.0*(np.sqrt(T*Tc))))

	return (-v**2)*(-rg*T/(v-bpr)**2 \
	              + 2.0*apr*alp(T)*(v+bpr)/(v**2+2.0*bpr*v-bpr**2)**2 \
	              - T/cvcalc_id_tro(T,ro)*(rg/(v-bpr)-dalpdT*apr/(v*v+2*v*bpr-bpr**2))**2)

#===============================================================================
def cpcalc_tro(T,ro):
#===============================================================================
#> Compute heat capacity at constant pressure
#> - PRS EOS -
#===============================================================================
	# specific volume
	v= 1.0/ro

	dalpdT= 2.0*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))*(-Kpr/(2.0*(np.sqrt(T*Tc))))

	d2alpdT2=(Kpr**2)/(2.0*Tc*T)+Kpr/(4.0*np.sqrt(Tc*T**3))*2.0*(1+Kpr*(1.0-np.sqrt(T/Tc)))

	cv= cvinf*abs(T/Tc)**nexp + apr*d2alpdT2*T/(bpr*np.sqrt(8)) \
	     *np.log(abs(2.0*v+2.0*bpr+bpr*np.sqrt(8))/abs(2.0*v+2.0*bpr-bpr*np.sqrt(8)))

	den=1.0/(v**2+2.0*bpr*v-bpr**2)

	dpdv=-rg*T/(v-bpr)**2 + 2.0*apr*alp(T)*(v+bpr)*den**2

	dpdT= rg/(v-bpr)-(dalpdT)*apr*den

	return cv - T*dpdT**2/dpdv


#===============================================================================
def cvcalc_id_tro(T,ro):
#===============================================================================
#> Compute heat capacity at constant volume in the ideal (dilute) limit
#> - PRS EOS -
#===============================================================================
	return cvinf*abs(T/Tc)**nexp


#===============================================================================
def cvcalc_tro(T,ro):
#===============================================================================
#> Compute heat capacity at constant volume
#> - PRS EOS -
#===============================================================================
	# pc,Tc,roc,zc,rg,om,nexp,cvinf,Tb,apr,bpr,L0,L1,L2,L3,Kpr = define()
	# cv = de/dT

	# specific volume
	v = 1.0/ro

	d2alpdT2= (Kpr**2)/(2.0*Tc*T) + Kpr/(4.0*np.sqrt(Tc*T**3))*2.0*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))

	return cvinf*abs(T/Tc)**nexp + apr*d2alpdT2*T/(bpr*np.sqrt(8)) \
	     *np.log(abs(2.0*v+2.0*bpr+bpr*np.sqrt(8))/abs(2.0*v+2.0*bpr-bpr*np.sqrt(8)))


#===============================================================================
def dpdicalc_tro(T,ro):
#===============================================================================
#> Compute pressure derivative w.r.t temperature
#> - PRS EOS -
#===============================================================================
	# specific volume
	v = 1.0/ro

	dalpdT = 2.0*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))*(-Kpr/(2.0*(np.sqrt(T*Tc))))

	d2alpdT2 = (Kpr**2)/(2.0*Tc*T) + Kpr/(4.0*np.sqrt(Tc*T**3))*2.0*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))

	cv = cvinf*abs(T/Tc)**nexp + apr*d2alpdT2*T/(bpr*np.sqrt(8)) \
	     * np.log(abs(2.0*v+2.0*bpr+bpr*np.sqrt(8))/abs(2.0*v+2.0*bpr-bpr*np.sqrt(8)))

	dpdT = rg/(v-bpr)-dalpdT*apr/(v**2+2.0*bpr*v-bpr**2)

	return dpdT/cv


#===============================================================================
def dpdTcalc_tro(T,ro):
#===============================================================================
#> Compute pressure derivative w.r.t temperature
#> - PRS EOS -
#===============================================================================
	# specific volume
	v = 1.0/ro

	dalpdT = 2.0*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))*(-Kpr/(2.0*(np.sqrt(T*Tc))))

	return rg/(v-bpr)-dalpdT*apr/(v**2+2.0*bpr*v-bpr**2)


#===============================================================================
def dpdvcalc_tro(T,ro):
#===============================================================================
#> Compute first-order pressure derivative w.r.t volume
#> - PRS EOS -
#===============================================================================
	# specific volume
	v= 1.0/ro

	return -rg*T/(v-bpr)**2 \
	                + 2.0*apr*alp(T)*(v+bpr)/(v**2+2.0*bpr*v-bpr**2)**2

#===============================================================================
def d2pdv2calc_tro(T,ro):
#===============================================================================
#> Compute second-order pressure derivative w.r.t volume
#> - PRS EOS -
#===============================================================================
	# specific volume
	v= 1.0/ro

	return 2.0*rg*T/(v-bpr)**3 \
	                  -2.0*apr*alp(T)*(3.0*v**2+6.0*bpr*v+5*bpr**2) \
	                                         /(v**2+2.0*bpr*v-bpr**2)**3

#===============================================================================
def d3pdv3calc_tro(T,ro):
#===============================================================================
#> Compute third-order pressure derivative w.r.t volume
#> - PRS EOS -
#===============================================================================
	# specific volume
	v= 1.0/ro

	return -6.0*rg*T/(v-bpr)**4 \
	                  + 24.0*apr*alp(T)*(v+bpr)*(v**2+2.0*bpr*v+3.0*bpr**2) \
	                                          /(v**2+2.0*bpr*v-bpr**2)**4

#===============================================================================
def ecalc_pro(p,ro,Ttent):
#===============================================================================
#> Compute internal energy from p and rho
#> - PRS EOS -
#===============================================================================
	# specific volume
	v= 1.0 / ro

	# initial guess
	T= Ttent

	# Newton's algorithm
	# ------------------
	err=1
	tol=1e-5
	while err>tol:

		dalpdT=2.0*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))*(-Kpr/(2.0*(np.sqrt(T*Tc))))

		# def \ derivative
		den=1.0/(v**2+2.0*bpr*v-bpr**2)
		fn= rg*T/(v-bpr) - apr*alp(T)*den-p
		der_fn= rg/(v-bpr)-dalpdT*apr*den

		# update solution
		T1= T - fn/der_fn

		err= (abs(T1-T)/T).max()
		T=T1

	return ecalc_tro(T1,ro)

#===============================================================================
def hcalc_tro(T,ro):
#===============================================================================
#> Compute enthaply from T and rho
#> - PRS EOS -
#===============================================================================

    return ecalc_tro(T,ro) + pcalc_tro(T,ro)/ro

#===============================================================================
def dedrocalc_tro(T,ro):
#===============================================================================
#> Compute internal energy derivative w.r.t rho
#> - PRS EOS -
#===============================================================================
	dalpdT=2.0*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))*(-Kpr/(2.0*(np.sqrt(T*Tc))))

	return -apr/bpr/np.sqrt(8)*(alp(T)-T*dalpdT)* \
	                  (2.0*bpr+bpr*np.sqrt(8))/((2.0*bpr+bpr*np.sqrt(8))*ro+2)- \
	                  (2.0*bpr-bpr*np.sqrt(8))/((2.0*bpr-bpr*np.sqrt(8))*ro+2)

#===============================================================================
def dedTcalc_tro(T,ro):
#===============================================================================
#> Compute internal energy derivative w.r.t temperature
#> - PRS EOS -
#===============================================================================
	# specific volume
	v=1.0/ro

	# second derivative of alpha
	d2alpdT2= (-Kpr/np.sqrt(T*Tc))*((Kpr/(-2.0*np.sqrt(T*Tc)))-(1+Kpr*(1.0-np.sqrt(T/Tc)))/(2.0*T))

	return cvinf*(T/Tc)**nexp-apr/bpr/np.sqrt(8)*T*d2alpdT2* \
	                 np.log(abs(2.0*v+2.0*bpr+bpr*np.sqrt(8))/abs(2.0*v+2.0*bpr-bpr*np.sqrt(8)))


#===============================================================================
def ecalc_tro(T,ro):
#===============================================================================
#> Compute internal energy from T and rho
#> - PRS EOS -
#===============================================================================
	# de = cvinf dT + (T*dp/dt-p) dv
	# specific volume
	v= 1.0/ro

	dalpdT=2.0*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))*(-Kpr/(2.0*(np.sqrt(T*Tc))))

	return cvinf/(nexp+1.0)*(abs(T/Tc))**(nexp+1.0)*Tc    \
	             - apr*(alp(T)-T*dalpdT)* 1.0/(bpr*np.sqrt(8)) \
	             *np.log(abs(2.0*v+2.0*bpr+bpr*np.sqrt(8))/abs(2.0*v+2.0*bpr-bpr*np.sqrt(8)))


#===============================================================================
def gcalc_tro(T,ro):
#===============================================================================
#> Compute fundamental derivative of gas dynamics from T and ro
#> - PRS EOS -
#===============================================================================
	# specific volume
	v= 1.0/ro

	# alpha
	alpr=alp(T)

	# first derivative of alpha
	dalpdT= -Kpr*(1+Kpr*(1.0-np.sqrt(T/Tc)))/np.sqrt(T*Tc)

	# seonc derivative of alpha
	d2alpdT2= (-Kpr/np.sqrt(T*Tc))*((Kpr/(-2.0*np.sqrt(T*Tc)))-(1+Kpr*(1.0-np.sqrt(T/Tc)))/(2.0*T))

	# third derivative of alpha
	d3alpdT3= -Kpr/np.sqrt(Tc*T)*(Kpr/(4.0*np.sqrt(Tc*T**3)) - (-Kpr/(2.0*np.sqrt(T*Tc)))/T \
	        + 3.0*(1+Kpr*(1.0-np.sqrt(T/Tc)))/(4.0*T*T))

	den=1.0/(v**2+2.0*bpr*v-bpr**2)

	dpdT= rg/(v-bpr)-dalpdT*apr*den

	d2pdT2= -apr*d2alpdT2*den

	d2pdvdT=-rg/(v-bpr)**2+2.0*apr*dalpdT*(v+bpr)*den**2

	d2pdv2= 2.0*rg*T/(v-bpr)**3 \
	      - 2.0*apr*alpr*(3.0*v**2+6.0*bpr*v+5.0*bpr**2)*den**3

	cc=np.log(abs(2.0*v+2.0*bpr+bpr*np.sqrt(8))/abs(2.0*v+2.0*bpr-bpr*np.sqrt(8)))

	cv = cvinf*abs(T/Tc)**nexp + apr*d2alpdT2*T/(bpr*np.sqrt(8))*cc

	dcvdT= nexp*cvinf*abs(T/Tc)**(nexp-1)/Tc + apr*(d2alpdT2+d3alpdT3*T)*1.0/(bpr*np.sqrt(8))*cc

	dpdv=-rg*T/(v-bpr)**2 + 2.0*apr*alpr*(v+bpr)*den**2

	c2calc = v**2*(T/cv*dpdT**2 - dpdv)

	return 0.5*v**3/c2calc*( d2pdv2 - 3.0*T/cv*dpdT*d2pdvdT \
	             + (T/cv*dpdT)**2*(3.0*d2pdT2+dpdT/T*(1.0-T/cv*dcvdT)) )

#===============================================================================
def pcalc_roero(roe,ro,Ttent):
#===============================================================================
#> Compute pressure from rhoe and rho
#> - PRS EOS -
#===============================================================================
	# specific volume
	v= 1.0/ro

	# initial guess
	T= Ttent

	# Newton's algorithm
	# ------------------
	err=1
	tol=1e-5
	while err>tol:
		alpr= alp(T)
		dalpdT= 2.0*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))*(-Kpr/(2.0*(np.sqrt(T*Tc))))
		d2alpdT2= (Kpr**2)/(2.0*Tc*T) + Kpr/(4.0*np.sqrt(Tc*T**3))*2.0*(1+Kpr*(1.0-np.sqrt(T/Tc)))

		# def \ derivative
		cc=np.log(abs(2.0*v+2.0*bpr+bpr*np.sqrt(8))/abs(2.0*v+2.0*bpr-bpr*np.sqrt(8)))

		fn =  cvinf/(nexp+1.0)*(abs(T/Tc))**(nexp+1.0)*Tc \
		    - apr*(alpr - T*dalpdT)* 1.0/(bpr*np.sqrt(8))*cc - roe/ro

		der_fn = cvinf*abs(T/Tc)**nexp + apr*d2alpdT2*T/(bpr*np.sqrt(8))*cc

		# update solution
		T1= T - fn/der_fn

		err= (abs(T1-T)/T).max()
		T= T1

	return rg*T/(v-bpr)-apr*alpr/(v**2+2.0*bpr*v-bpr**2)



#===============================================================================
def pcalc_tro(T,ro):
#===============================================================================
#> Compute pressure from T and rho
#> - PRS EOS -
#===============================================================================
	# specific volume
	v= 1.0/ro

	return rg*T/(v-bpr) - apr*alp(T)/(v**2+2.0*v*bpr-bpr**2)


#===============================================================================
def rocalc_ep(e,p,Ttent):
#===============================================================================
#> Compute density from internal energy e and p
#> - PRS EOS -
#===============================================================================

    # initial guess
    T1= Ttent
    ro= 0.1*roc

    # iterations
    err=1
    tol=1e-5
    while err>tol:
        T = T1
        ro= rocalc_pt(p,T,ro)
        T1= tcalc_roero(ro*e,ro,T1)
        err= abs(T1-T)/T

        rocalc_ep= ro

    return rocalc_ep


#===============================================================================
def rocalc_ps(p,s,ro,Ttent):
#===============================================================================
#> Compute density from p and entropy s
#> - PRS EOS -
#===============================================================================
	# initial guess
	T1= Ttent
	ro= 0.1*roc

	# iterations
	err=1
	tol=1e-5
	while err>tol:
		T = T1
		T1= tcalc_sro(s,ro,T1)
		err= (abs(T1-T)/T).max()

	return ro

#===============================================================================
def rocalc_pt(p,T,rotent):
#===============================================================================
#> Compute density from p and T
#> - PRS EOS -
#===============================================================================
	# initial guess (specific volume)
	v= 1.0/rotent

	# Newton's algorithm
	# ------------------
	err=1
	tol=1e-5
	while err>tol:

		# def \ derivative
		fn = rg*T/(v-bpr)- apr*alp(T)/(v**2+2.0*bpr*v-bpr**2) - p
		der_fn = -rg*T/(v-bpr)**2 + 2.0*apr*alp(T)*(v+bpr)/(v**2+2.0*v*bpr-bpr**2)**2

		# update solution
		v1 = v - fn/der_fn

		err = (abs(v1-v)/v).max()
		v=v1

	return 1.0/v1

#===============================================================================
def rocalc_st(s,T,rotent):
#===============================================================================
#> Compute density from entropy and T
#> - PRS EOS -
#===============================================================================
	dalpdT = 2.0*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))*(-Kpr/(2.0*(np.sqrt(T*Tc))))

	# initial guess (specific volume)
	v = 1.0/rotent

	# Newton's algorithm
	# ------------------
	err=1
	tol=1e-5
	while err>tol:

		# def \ derivative
		fn = cvinf/nexp*abs(T/Tc)**nexp + rg*np.log(v-bpr)+ apr*dalpdT* 1.0/(bpr*np.sqrt(8))\
		*np.log(abs(2.0*v+2.0*bpr+bpr*np.sqrt(8))/abs(2.0*v+2.0*bpr-bpr*np.sqrt(8))) - s
		der_fn = rg/(v-bpr)-apr*dalpdT/(v**2+2.0*bpr*v-bpr**2)

		# update solution
		v1= v - fn/der_fn

		err = (abs(v1-v)/v).max()
		v = v1

	return 1.0/v1

#===============================================================================
def scalc_tro(T,ro):
#===============================================================================
#> Compute entropy s from T and rho
#> - PRS EOS -
#===============================================================================
	# ds = de/T + pdv/T
	# specific volume
	v= 1.0/ro

	dalpdT = 2.0*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))*(-Kpr/(2.0*(np.sqrt(T*Tc))))

	return cvinf/nexp*abs(T/Tc)**nexp + rg*np.log(v-bpr)+ apr*dalpdT* 1.0/(bpr*np.sqrt(8))\
	     * np.log(abs(2.0*v+2.0*bpr+bpr*np.sqrt(8))/abs(2.0*v+2.0*bpr-bpr*np.sqrt(8)))


#===============================================================================
def tcalc_ph(p,h,rotent,Ttent):
#===============================================================================
#> Compute temperature from p and enthalpy h
#> - PRS EOS - 
#===============================================================================
	# initial guess
	T1= Ttent

	# iterations
	err=1
	tol=1e-5
	while err>tol:
		T  = T1
		ro = rocalc_pt(p,T,rotent)

		e = h-p/ro
		T1= tcalc_roero(ro*e,ro,T)
		err= (abs(T1-T)/T).max()

	return T1
 

#===============================================================================
def tcalc_pro(p,ro,Ttent):
#===============================================================================
#> Compute temperature from p and rho
#> - PRS EOS - 
# Tn+1 = Tn -(P(Tn)-Pref)/(dPdT(Tn))
#===============================================================================
	# specific volume
	v= 1.0 / ro

	# initial guess
	T = Ttent

	# iterations
	err=1
	tol=1e-5
	while err>tol:

		dalpdT= -Kpr*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))/np.sqrt(T*Tc)

		T1= T - (rg*T/(v-bpr)-apr*alp(T)/(v**2+2.0*bpr*v-bpr**2) - p) \
		      /(rg/(v-bpr)-dalpdT*apr/(v**2 +2.0*bpr*v-bpr**2))

		err= (abs(T1-T)/T).max()
		T=T1

	return T1

#===============================================================================
def tcalc_roero(roe,ro,Ttent):
#===============================================================================
#> Compute temperature from rho*e and rho
#> - PRS EOS -
#===============================================================================
	# specific volume
	v= 1.0/ro
	cc=np.log(abs(2.0*v+2.0*bpr+bpr*np.sqrt(8))/abs(2.0*v+2.0*bpr-bpr*np.sqrt(8)))

	# initial guess
	T= Ttent

	# Newton's algorithm
	# ------------------
	err=1
	tol=1e-3
	while err>tol:

		dalpdT = 2.0*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))*(-Kpr/(2.0*(np.sqrt(T*Tc))))
		d2alpdT2 = (Kpr**2)/(2.0*Tc*T) + Kpr/(4.0*np.sqrt(Tc*T**3))*2.0*(1+Kpr*(1.0-np.sqrt(T/Tc)))

		# def \ derivative       
		fn= cvinf/(nexp+1.0)*(abs(T/Tc))**(nexp+1.0)*Tc \
		 - apr*(alp(T)-T*dalpdT)/(bpr*np.sqrt(8))*cc  - roe/ro

		der_fn= cvinf*abs(T/Tc)**nexp + apr*d2alpdT2*T/(bpr*np.sqrt(8))*cc

		# update solution
		T1= T - fn/der_fn

		err= (abs(T1-T)/T).max()
		T=T1

	return T1

#===============================================================================
def tcalc_sro(s,ro,Ttent):
#===============================================================================
#> Compute temperature from entropy s and rho
#> - PRS EOS -
#===============================================================================
	# specific volume
	v= 1.0/ro

	T= Ttent

	# Newton's algorithm
	# ------------------
	err=1
	tol=1e-5
	while err>tol:

		dalpdT= 2.0*(1.0+Kpr*(1.0-np.sqrt(T/Tc)))*(-Kpr/(2.0*(np.sqrt(T*Tc))))

		d2alpdT2= (Kpr**2)/(2.0*Tc*T) + Kpr/(4.0*np.sqrt(Tc*T**3))*2.0*(1+Kpr*(1.0-np.sqrt(T/Tc)))

		# def \ derivative
		cc=np.log(abs(2.0*v+2.0*bpr+bpr*np.sqrt(8))/abs(2.0*v+2.0*bpr-bpr*np.sqrt(8)))

		fn= cvinf/nexp*abs(T/Tc)**nexp+rg*np.log(v-bpr)+apr*dalpdT/(bpr*np.sqrt(8))*cc - s
		der_fn= cvinf*abs(T/Tc)**(nexp-1)/Tc + apr*d2alpdT2 * 1.0/(bpr*np.sqrt(8))*cc

		# update solution
		T1= T - fn/der_fn

		err= (abs(T1-T)/T).max()
		T=T1

	return T1

#===============================================================================
def vvol(pi,Ti,vtent):
#===============================================================================
#> Compute vvol such that p(vvol)=p
#> - PRS EOS -
#===============================================================================
	T= Ti*Tc
	p= pi*pc

	# initial guess (specific volume)
	v0 = vtent/roc

	# iterations
	err=1
	tol=1e-5
	while err>tol:
		v1 = v0 - (p-pcalc_tro(T,1.0/v0))/(-dpdvcalc_tro(T,1.0/v0))

		err = (abs(v1-v0)/v0).max()
		v0=v1

	return v1*roc

#===============================================================================
def vvol_d1(Ti,vtent):
#===============================================================================
#> Compute vvol_d1 from Ti and tentative v
#> - PRS EOS -
#===============================================================================
	T= Ti*Tc

	# initial guess (specific volume)
	v0 = vtent/roc

	# iterations
	err=1
	tol=1e-5
	for i in range(15000):
		v1=v0-dpdvcalc_tro(T,1.0/v0)/(d2pdv2calc_tro(T,1.0/v0)+1.e-16)

		err = (abs(v1-v0)/v0).max()
		if err<=tol1:
			return v1*roc

#===============================================================================
def vvol_d2(Ti,vtent):
#===============================================================================
#> Compute vvol_d2 from Ti and tentative v
#> - PRS EOS -
#===============================================================================
	T = Ti*Tc
	# initial guess (specific volume)
	v0 = vtent/roc

	# iterations
	err=1
	tol=1e-5
	while err>tol:
		v1= v0-d2pdv2calc_tro(T,1.0/v0)/(d3pdv3calc_tro(T,1.0/v0)+1.e-16)

		err= (abs(v1-v0)/v0).max()
		v0=v1

	return v1*roc

#===============================================================================
def intpcalc_tro(Ti,roi):
#===============================================================================
#> Compute pressure integral from T and rho
#> - PRS EOS -
#===============================================================================
	T = Ti*Tc
	ro = roi*roc

	# specific volume
	v= 1.0/ro

	temp= rg*T*np.log(v-bpr) + apr*alp(T)/(bpr*np.sqrt(8))\
	     *np.log(abs(2.0*v+2.0*bpr+bpr*np.sqrt(8))/abs(2.0*v+2.0*bpr-bpr*np.sqrt(8)))

	return temp/pc*roc
