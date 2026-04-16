def interpolate_profile(profile, original_profile, kind='linear'):
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
    new_upper, new_lower, _ = split_airfoil(profile_scaled)

    # Separa il profilo originale (che fornisce i nodi di interpolazione)
    orig_upper, orig_lower, orig_split_indices = split_airfoil(original_profile)

    # OTTIMIZZAZIONE: Crea le funzioni di interpolazione una sola volta invece che ad ogni iterazione
    # Prepara le superfici ordinate per x
    new_upper_sorted = new_upper[np.argsort(new_upper[:, 0])]
    new_lower_sorted = new_lower[np.argsort(new_lower[:, 0])]
    orig_upper_sorted = orig_upper[np.argsort(orig_upper[:, 0])]
    orig_lower_sorted = orig_lower[np.argsort(orig_lower[:, 0])]

    # Crea funzioni di interpolazione (una volta sola)
    f_new_upper = interp1d(new_upper_sorted[:, 0], new_upper_sorted[:, 1], kind=kind, bounds_error=False, fill_value='extrapolate')
    f_new_lower = interp1d(new_lower_sorted[:, 0], new_lower_sorted[:, 1], kind=kind, bounds_error=False, fill_value='extrapolate')
    f_orig_upper = interp1d(orig_upper_sorted[:, 0], orig_upper_sorted[:, 1], kind=kind, bounds_error=False, fill_value='extrapolate')
    f_orig_lower = interp1d(orig_lower_sorted[:, 0], orig_lower_sorted[:, 1], kind=kind, bounds_error=False, fill_value='extrapolate')

    # Crea un profilo interpolato con le stesse coordinate x del profilo originale
    interpolated = np.zeros_like(original_profile)
    interpolated[:, 0] = original_profile[:, 0]  # Copia esattamente le coordinate x

    # OTTIMIZZAZIONE: Interpola tutti i punti in una volta (vectorizzato)
    x_coords = original_profile[:, 0]
    y_coords_orig = original_profile[:, 1]

    # Calcola y su entrambe le superfici per tutti i punti
    y_upper_orig = f_orig_upper(x_coords)
    y_lower_orig = f_orig_lower(x_coords)
    y_upper_new = f_new_upper(x_coords)
    y_lower_new = f_new_lower(x_coords)

    # Determina per ogni punto se appartiene a dorso o ventre (operazione vectorizzata)
    dist_to_upper = np.abs(y_coords_orig - y_upper_orig)
    dist_to_lower = np.abs(y_coords_orig - y_lower_orig)
    is_upper = dist_to_upper < dist_to_lower

    # Assegna i valori interpolati (operazione vectorizzata)
    interpolated[:, 1] = np.where(is_upper, y_upper_new, y_lower_new)

    # Post-processing: correggi punti vicini al trailing edge che sono stati classificati male
    # OTTIMIZZAZIONE: Calcola distanze in modo vectorizzato
    distances = np.sqrt(np.diff(interpolated[:, 0])**2 + np.diff(interpolated[:, 1])**2)
    mean_distance = np.mean(distances)
    threshold = mean_distance * 0.05

    # Identifica coppie di punti consecutivi troppo vicini
    close_pairs = np.where(distances < threshold)[0]
    # print(len(close_pairs), " coppie di punti vicini trovate per correzione")

    # Se non ci sono coppie problematiche, restituisci subito
    if len(close_pairs) == 0:
        return interpolated

    # Filtra solo le coppie vicine agli split indices
    near_split_mask = np.array([
        (abs(idx - orig_split_indices[0]) < 3 or
         abs(idx - orig_split_indices[1]) < 3 or
         abs(idx + 1 - orig_split_indices[0]) < 3 or
         abs(idx + 1 - orig_split_indices[1]) < 3)
        for idx in close_pairs
    ])
    close_pairs = close_pairs[near_split_mask]

    # Se non ci sono coppie problematiche vicino agli split, restituisci subito
    if len(close_pairs) == 0:
        return interpolated

    # Per le coppie rimanenti, applica la correzione (necessario loop per has_intersection)
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

    # Precalcola tutti i valori y necessari per le correzioni (vectorizzato)
    x_vals = interpolated[close_pairs, 0]
    x_vals_next = interpolated[close_pairs + 1, 0]

    y_upper_vals = f_new_upper(x_vals)
    y_lower_vals = f_new_lower(x_vals)
    y_upper_vals_next = f_new_upper(x_vals_next)
    y_lower_vals_next = f_new_lower(x_vals_next)

    for pair_idx, idx in enumerate(close_pairs):
        i = idx
        j = idx + 1

        y_i_upper = y_upper_vals[pair_idx]
        y_i_lower = y_lower_vals[pair_idx]
        y_j_upper = y_upper_vals_next[pair_idx]
        y_j_lower = y_lower_vals_next[pair_idx]

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

    return interpolated
