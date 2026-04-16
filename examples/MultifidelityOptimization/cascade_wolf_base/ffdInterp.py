def interpolate_profile(self, profile, original_profile, kind='linear'):
    """
    Interpola un profilo generico usando i nodi definiti da un profilo originale.
    Separa entrambi i profili in dorso e ventre, poi interpola ciascuna superficie
    sui nodi corrispondenti del profilo originale.

    Args:
        profile: 2D array (x, y) del profilo da interpolare (nuovo profilo)
        original_profile: 2D array (x, y) del profilo che definisce i nodi di interpolazione
        kind: tipo di interpolazione ('linear', 'cubic', etc.)

    Returns:
        2D array con profilo interpolato che mantiene esattamente le coordinate x del profilo originale
    """

    # Calcola min e max del profilo originale
    x_min_orig = np.min(original_profile[:, 0])
    x_max_orig = np.max(original_profile[:, 0])

    # Scala il profilo nuovo per matchare il range del profilo originale
    x_min_new = np.min(profile[:, 0])
    x_max_new = np.max(profile[:, 0])

    # Crea una copia del profilo da scalare
    profile_scaled = profile.copy()

    # Scala le coordinate x del nuovo profilo per matchare il range dell'originale
    if x_max_new != x_min_new:  # Evita divisione per zero
        profile_scaled[:, 0] = (profile[:, 0] - x_min_new) / (x_max_new - x_min_new) * (x_max_orig - x_min_orig) + x_min_orig

    # Separa il profilo nuovo (da interpolare) dopo lo scaling
    new_upper, new_lower, _ = self.split_airfoil(profile_scaled)

    # Separa il profilo originale (che fornisce i nodi di interpolazione)
    _, _, orig_split_indices = self.split_airfoil(original_profile)

    # Crea un profilo interpolato con le stesse coordinate x del profilo originale
    interpolated = np.zeros_like(original_profile)
    interpolated[:, 0] = original_profile[:, 0]  # Copia esattamente le coordinate x

    # Per ogni punto del profilo originale, determina se appartiene a dorso o ventre
    # e interpola usando la superficie corretta
    for i in range(len(original_profile)):
        x_target = original_profile[i, 0]

        # Determina se il punto corrente è sul dorso o sul ventre
        # confrontando con gli indici di split del profilo originale
        if i <= orig_split_indices[0] or i >= orig_split_indices[1]:
            # Punto sul ventre (lower)
            y_interp = self.interp_surface(new_lower, np.array([x_target]), kind)[0, 1]
        else:
            # Punto sul dorso (upper)
            y_interp = self.interp_surface(new_upper, np.array([x_target]), kind)[0, 1]

        interpolated[i, 1] = y_interp

    # Post-processing: correggi punti vicini al trailing edge che sono stati classificati male
    # Calcola la distanza euclidea tra punti consecutivi
    distances = np.sqrt(np.diff(interpolated[:, 0])**2 + np.diff(interpolated[:, 1])**2)
    mean_distance = np.mean(distances)

    # Soglia per identificare punti "troppo vicini" (distanza molto minore della media)
    threshold = mean_distance * 0.05  # Punti con distanza < 15% della media

    # Identifica coppie di punti consecutivi troppo vicini
    close_pairs = np.where(distances < threshold)[0]

    corrected_points = 0
    for idx in close_pairs:
        i = idx  # Primo punto della coppia
        j = idx + 1  # Secondo punto della coppia

        # Verifica che siano vicini a una zona di transizione (split_indices)
        # Considera una finestra intorno agli split indices
        near_split = (abs(i - orig_split_indices[0]) < 3 or
                    abs(i - orig_split_indices[1]) < 3 or
                    abs(j - orig_split_indices[0]) < 3 or
                    abs(j - orig_split_indices[1]) < 3)

        if near_split:
            x_i = interpolated[i, 0]
            x_j = interpolated[j, 0]

            y_i_upper = self.interp_surface(new_upper, np.array([x_i]), kind)[0, 1]
            y_i_lower = self.interp_surface(new_lower, np.array([x_i]), kind)[0, 1]
            y_j_upper = self.interp_surface(new_upper, np.array([x_j]), kind)[0, 1]
            y_j_lower = self.interp_surface(new_lower, np.array([x_j]), kind)[0, 1]

            # Test auto-intersezione: verifica se segmenti (p1,p2) e (p3,p4) si intersecano
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

            def has_intersection(prof, idx1, idx2):
                """Verifica se modificare i,j crea intersezioni con altri segmenti"""
                segs = [(idx1-1, idx1), (idx1, idx2), (idx2, idx2+1)]
                segs = [(a, b) for a, b in segs if 0 <= a < len(prof) and 0 <= b < len(prof)]
                for a, b in segs:
                    p1, p2 = prof[a], prof[b]
                    for k in range(len(prof) - 1):
                        if abs(k - a) <= 1 or abs(k - b) <= 1:
                            continue
                        p3, p4 = prof[k], prof[k + 1]
                        if ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4):
                            return True
                return False

            # Testa configurazioni alternative
            configs = [
                (y_i_upper, y_j_lower, abs(y_i_upper - y_j_lower)),
                (y_i_lower, y_j_upper, abs(y_i_lower - y_j_upper))
            ]

            current_dist = abs(interpolated[i, 1] - interpolated[j, 1])
            best_config = None
            best_dist = current_dist

            for yi, yj, dist in configs:
                test_prof = interpolated.copy()
                test_prof[i, 1], test_prof[j, 1] = yi, yj
                if not has_intersection(test_prof, i, j) and dist > best_dist * 1.5:
                    best_config = (yi, yj)
                    best_dist = dist

            if best_config:
                interpolated[i, 1], interpolated[j, 1] = best_config
                corrected_points += 2

