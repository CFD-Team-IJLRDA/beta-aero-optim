###################

# Valutazione spline
tck, u = splprep([xProfile, yProfile], s=0, k=2, per = True)

# Calcolo normali (rispetto alla curva originale xProfile, yProfile)
dx, dy = splev(u, tck, der=1)
norm = np.sqrt(dx**2 + dy**2)
tx, ty = dx/norm, dy/norm
nx, ny = -dy/norm, dx/norm

u_fine = np.linspace(0, 1, 10000)
x_s, y_s = splev(u_fine, tck)
# Plot
plt.figure(figsize=(8, 6))
plt.plot(xProfile[10:20], yProfile[10:20], '.', label='Punti')
plt.plot(x_s, y_s, '-', label='Spline')
plt.quiver(xProfile[10:20], yProfile[10:20], nx[10:20], ny[10:20], color='red', label='Normali')
plt.quiver(xProfile[10:20], yProfile[10:20], ny[10:20], -nx[10:20], color='green', label='Normali')
plt.axis('equal')
plt.show()

tau[:,0,0] = mu*2*dudx
tau[:,0,1] = mu*(dudy+dvdx)
tau[:,1,0] = mu*(dudy+dvdx)
tau[:,1,1] = mu*2*dvdy

tauw = []
for idx in range(len(tau)):
    val = (tau[idx,0,0]*nx[idx]+tau[idx,0,1]*ny[idx])*tx[idx] + (tau[idx,1,0]*nx[idx]+tau[idx,1,1]*ny[idx])*ty[idx]
    tauw.append(val)

cf = np.array(tauw)/(0.5*rho_in*(u_in**2))

##################
