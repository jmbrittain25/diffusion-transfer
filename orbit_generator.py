import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from astropy.constants import G, M_earth
from astropy.coordinates import (
    GCRS, ITRS, CartesianRepresentation, CartesianDifferential
)
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver
from tqdm import tqdm
import multiprocessing
from functools import partial

# Constants
MU = (G * M_earth).to(u.km**3 / u.s**2).value
R_EARTH = 6371.0
EPS = 1e-12  # Small epsilon for divisions

def cartesian_to_spherical(pos, vel):
    """Convert [x,y,z] pos, [vx,vy,vz] vel to [r, theta_deg, phi_deg, vr, vtheta, vphi]."""
    x, y, z = pos
    vx, vy, vz = vel
    r = np.sqrt(x**2 + y**2 + z**2 + EPS)
    theta = np.degrees(np.arccos(z / r))
    phi = np.degrees(np.arctan2(y, x))
    vr = (x*vx + y*vy + z*vz) / r
    sin_theta = np.sin(np.radians(theta))
    vtheta = 0.0
    vphi = 0.0
    if sin_theta > EPS:
        vtheta = ((x*vx + y*vy) * np.cos(np.radians(theta)) - z * vr) / (r * sin_theta)
        vphi = (x*vy - y*vx) / (r**2 * sin_theta)
    return np.array([r, theta, phi, vr, vtheta, vphi])

def generate_random_orbit(epoch, orbit_type="Random", two_d=False):
    """Generate random Poliastro Orbit."""
    if orbit_type == "Random":
        orbit_type = np.random.choice(["LEO", "MEO", "HEO", "GEO"])

    if orbit_type == "LEO":
        periapsis_alt_min, periapsis_alt_max = 160, 2000
        apoapsis_alt_min, apoapsis_alt_max = 160, 2000
    elif orbit_type == "MEO":
        periapsis_alt_min, periapsis_alt_max = 2000, 35786
        apoapsis_alt_min, apoapsis_alt_max = 2000, 35786
    elif orbit_type == "HEO":
        periapsis_alt_min, periapsis_alt_max = 160, 2000
        apoapsis_alt_min, apoapsis_alt_max = 2000, 35786
    elif orbit_type == "GEO":
        periapsis_alt_min, periapsis_alt_max = 35786, 35786
        apoapsis_alt_min, apoapsis_alt_max = 35786, 35786
    else:
        raise ValueError("Invalid orbit type.")

    periapsis_alt = np.random.uniform(periapsis_alt_min, periapsis_alt_max)
    apoapsis_alt = np.random.uniform(apoapsis_alt_min, apoapsis_alt_max)
    while apoapsis_alt < periapsis_alt:
        apoapsis_alt = np.random.uniform(apoapsis_alt_min, apoapsis_alt_max)

    periapsis = (R_EARTH + periapsis_alt) * u.km
    apoapsis = (R_EARTH + apoapsis_alt) * u.km
    a = (periapsis + apoapsis) / 2
    ecc = (apoapsis - periapsis) / (apoapsis + periapsis)

    inc = 0 * u.deg if two_d else np.random.uniform(0, 180) * u.deg
    raan = 0 * u.deg if two_d else np.random.uniform(0, 360) * u.deg
    argp = np.random.uniform(0, 360) * u.deg
    nu = np.random.uniform(0, 360) * u.deg

    orb = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu, epoch=epoch)
    return orb, orbit_type

def propagate_orbit_to_df(orb, orbit_id, orbit_regime, time_step=60*u.s, num_steps=None, base_epoch=None):
    """Propagate Orbit and build DF."""
    if base_epoch is None:
        base_epoch = orb.epoch

    period = orb.period
    if num_steps is None:
        num_steps = int(period.to_value(u.s) // time_step.to_value(u.s)) - 1
    else:
        time_step = period / num_steps

    rows = []
    for step in range(num_steps):
        dt = step * time_step
        new_orb = orb.propagate(dt)

        current_time = base_epoch + dt
        r_eci = new_orb.r.to_value(u.km)
        v_eci = new_orb.v.to_value(u.km / u.s)
        x_eci, y_eci, z_eci = r_eci
        vx_eci, vy_eci, vz_eci = v_eci

        gcrs_coord = GCRS(
            x=x_eci*u.km, y=y_eci*u.km, z=z_eci*u.km,
            v_x=vx_eci*(u.km/u.s), v_y=vy_eci*(u.km/u.s), v_z=vz_eci*(u.km/u.s),
            representation_type=CartesianRepresentation,
            differential_type=CartesianDifferential,
            obstime=current_time
        )
        itrs_coord = gcrs_coord.transform_to(ITRS(obstime=current_time))
        x_ecef, y_ecef, z_ecef = itrs_coord.x.to_value(u.km), itrs_coord.y.to_value(u.km), itrs_coord.z.to_value(u.km)
        vx_ecef, vy_ecef, vz_ecef = itrs_coord.v_x.to_value(u.km/u.s), itrs_coord.v_y.to_value(u.km/u.s), itrs_coord.v_z.to_value(u.km/u.s)

        sph_eci = cartesian_to_spherical(r_eci, v_eci)
        sph_ecef = cartesian_to_spherical([x_ecef, y_ecef, z_ecef], [vx_ecef, vy_ecef, vz_ecef])
        r_eci_val, theta_eci_val, phi_eci_val, vr_eci, vtheta_eci, vphi_eci = sph_eci
        r_ecef_val, theta_ecef_val, phi_ecef_val, vr_ecef, vtheta_ecef, vphi_ecef = sph_ecef

        row = {
            "orbit_id": orbit_id,
            "orbit_regime": orbit_regime,
            "epoch": base_epoch.isot,
            "period_s": period.to_value(u.s),
            "time_s": dt.to_value(u.s),
            "x_eci_km": x_eci, "y_eci_km": y_eci, "z_eci_km": z_eci,
            "vx_eci_km_s": vx_eci, "vy_eci_km_s": vy_eci, "vz_eci_km_s": vz_eci,
            "r_eci_km": r_eci_val, "theta_eci_deg": theta_eci_val, "phi_eci_deg": phi_eci_val,
            "vr_eci_km_s": vr_eci, "vtheta_eci_km_s": vtheta_eci, "vphi_eci_km_s": vphi_eci,
            "x_ecef_km": x_ecef, "y_ecef_km": y_ecef, "z_ecef_km": z_ecef,
            "vx_ecef_km_s": vx_ecef, "vy_ecef_km_s": vy_ecef, "vz_ecef_km_s": vz_ecef,
            "r_ecef_km": r_ecef_val, "theta_ecef_deg": theta_ecef_val, "phi_ecef_deg": phi_ecef_val,
            "vr_ecef_km_s": vr_ecef, "vtheta_ecef_km_s": vtheta_ecef, "vphi_ecef_km_s": vphi_ecef,
            "sma_km": new_orb.a.to_value(u.km), "ecc": new_orb.ecc.value,
            "inc_deg": new_orb.inc.to_value(u.deg), "raan_deg": new_orb.raan.to_value(u.deg),
            "argp_deg": new_orb.argp.to_value(u.deg), "nu_deg": new_orb.nu.to_value(u.deg),
        }
        rows.append(row)

    return pd.DataFrame(rows)

def generate_single_orbit(i, orbit_types, time_step, num_steps, base_epoch, two_d=False):
    np.random.seed(i)
    chosen_type = np.random.choice(orbit_types)
    orb, _ = generate_random_orbit(base_epoch, chosen_type, two_d)
    orbit_id = f"Orbit_{i+1}"
    df = propagate_orbit_to_df(orb, orbit_id, chosen_type, time_step, num_steps, base_epoch)
    return df

def generate_orbits_dataset(n_orbits=2, orbit_types=("LEO", "MEO", "HEO", "GEO"), time_step=60*u.s, num_steps=None,
                            out_csv=None, out_npz=None, num_workers=None, two_d=False):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    base_epoch = Time("J2000", scale="tt")
    generate_func = partial(generate_single_orbit, orbit_types=orbit_types, time_step=time_step, num_steps=num_steps, base_epoch=base_epoch, two_d=two_d)
    
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(generate_func, range(n_orbits)), total=n_orbits, desc="Generating orbits"))
    
    final_df = pd.concat(results, ignore_index=True)
    if out_csv:
        final_df.to_csv(out_csv, index=False)
        print(f"Saved {len(final_df)} rows to {out_csv}")
    
    if out_npz:
        trajectories = []
        types = []
        for df in results:
            cols = ['x_eci_km', 'y_eci_km', 'vx_eci_km_s', 'vy_eci_km_s', 'time_s'] if two_d else ['x_eci_km', 'y_eci_km', 'z_eci_km', 'vx_eci_km_s', 'vy_eci_km_s', 'vz_eci_km_s', 'time_s']
            traj = df[cols].to_numpy()
            t_norm = traj[:, -1] / traj[-1, -1] if len(traj) > 0 and traj[-1, -1] > 0 else np.zeros(len(traj))
            traj[:, -1] = t_norm
            trajectories.append(traj)
            types.append(df['orbit_regime'].iloc[0])
        np.savez(out_npz, trajectories=np.array(trajectories, dtype=object), types=np.array(types))
        print(f"Saved NPZ to {out_npz}")
    
    return final_df

# TODO - implement safety check for reasonable orbits!
def generate_perturbed_trajectory(i, max_dv_per=0.5, max_num_burns=3, max_total_dv=1.0, 
                                  max_time_btwn=3600*u.s, max_total_time=86400*u.s, time_step=60*u.s, 
                                  num_steps_per_seg=None, two_d=False):
    np.random.seed(int(i))
    orbit_id = f"Orbit_{i+1}"
    base_epoch = Time("J2000", scale="tt")
    orb, orbit_type = generate_random_orbit(base_epoch, two_d=two_d)

    num_burns = np.random.randint(1, max_num_burns + 1)
    total_dv_used = 0.0
    current_orb = orb
    current_time = 0.0 * u.s
    dfs = []

    for burn_num in range(num_burns):
        time_to_next = np.random.uniform(time_step.to_value(u.s), max_time_btwn.to_value(u.s)) * u.s
        if current_time + time_to_next > max_total_time:
            time_to_next = max_total_time - current_time
        seg_steps = num_steps_per_seg or int(time_to_next.to_value(u.s) // time_step.to_value(u.s))
        seg_df = propagate_orbit_to_df(current_orb, orbit_id, orbit_type, time_to_next / seg_steps, seg_steps, current_orb.epoch)
        dfs.append(seg_df)

        # Apply tangential burn
        v_arr = current_orb.v.to_value(u.km / u.s)
        v_dir = v_arr / np.linalg.norm(v_arr)
        dv_mag = np.random.uniform(-max_dv_per, max_dv_per)
        if abs(total_dv_used + dv_mag) > max_total_dv:
            dv_mag = np.sign(dv_mag) * (max_total_dv - abs(total_dv_used))

        # NOTE - changed to meters per second!!
        dv_vec = (dv_mag * u.m / u.s) * v_dir
        
        maneuver = Maneuver.impulse(dv_vec)
        current_orb = current_orb.apply_maneuver(maneuver)
        total_dv_used += abs(dv_mag)
        current_time += time_to_next

        if current_time >= max_total_time or total_dv_used >= max_total_dv:
            break

    # Final segment
    if current_time < max_total_time:
        remaining_time = max_total_time - current_time
        final_steps = num_steps_per_seg or int(remaining_time.to_value(u.s) // time_step.to_value(u.s))
        final_df = propagate_orbit_to_df(current_orb, orbit_id, orbit_type, remaining_time / final_steps, final_steps, current_orb.epoch)
        dfs.append(final_df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df['time_s'] = np.cumsum(full_df['time_s'].diff().fillna(full_df['time_s'].iloc[0]))
    return full_df, orbit_type

def generate_perturbed_dataset(n_orbits=1000, max_dv_per=0.5, max_num_burns=3, max_total_dv=1.0,
                               max_time_btwn=3600*u.s, max_total_time=86400*u.s, time_step=60*u.s,
                               num_steps_per_seg=None, out_csv=None, out_npz=None, num_workers=None, two_d=False):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    generate_func = partial(generate_perturbed_trajectory, max_dv_per=max_dv_per, max_num_burns=max_num_burns, max_total_dv=max_total_dv,
                            max_time_btwn=max_time_btwn, max_total_time=max_total_time, time_step=time_step, num_steps_per_seg=num_steps_per_seg, two_d=two_d)
    
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(generate_func, range(n_orbits)), total=n_orbits, desc="Generating perturbed orbits"))
    
    dfs, types = zip(*results)
    final_df = pd.concat(dfs, ignore_index=True)
    if out_csv:
        final_df.to_csv(out_csv, index=False)
        print(f"Saved {len(final_df)} rows to {out_csv}")
    
    if out_npz:
        trajectories = []
        for df in dfs:
            cols = ['x_eci_km', 'y_eci_km', 'vx_eci_km_s', 'vy_eci_km_s', 'time_s'] if two_d else ['x_eci_km', 'y_eci_km', 'z_eci_km', 'vx_eci_km_s', 'vy_eci_km_s', 'vz_eci_km_s', 'time_s']
            traj = df[cols].to_numpy()
            t_norm = traj[:, -1] / traj[-1, -1] if len(traj) > 0 and traj[-1, -1] > 0 else np.zeros(len(traj))
            traj[:, -1] = t_norm
            trajectories.append(traj)
        np.savez(out_npz, trajectories=np.array(trajectories, dtype=object), types=np.array(types))
        print(f"Saved NPZ to {out_npz}")
    
    return final_df

def split_orbits_by_id(df, train_ratio=0.8, val_ratio=0.1):
    unique_ids = df["orbit_id"].unique()
    np.random.shuffle(unique_ids)
    n_total = len(unique_ids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val
    train_ids = unique_ids[:n_train]
    val_ids = unique_ids[n_train:n_train + n_val]
    test_ids = unique_ids[n_train + n_val:]
    df_train = df[df["orbit_id"].isin(train_ids)].copy()
    df_val = df[df["orbit_id"].isin(val_ids)].copy()
    df_test = df[df["orbit_id"].isin(test_ids)].copy()
    return df_train, df_val, df_test
