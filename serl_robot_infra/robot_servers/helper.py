import torch
from scipy.spatial.transform import Rotation as R









def pseudo_inverse(M : torch.Tensor, damped=True):
    lambda_ = 0.2 if damped else 0.0

    # Perform SVD
    U, s, Vt = torch.linalg.svd(M, full_matrices=True)
    
    # Create the diagonal matrix S
    S = torch.zeros_like(M.T)
    
    # Compute the pseudoinverse of the singular values
    for i in range(len(s)):
        S[i, i] = s[i] / (s[i]**2 + lambda_**2)
     #TODO size mismatch
    # Compute the pseudoinverse
     #M_pinv = Vt @ S.T @ U.T
    
    return torch.linalg.pinv(M)

def saturate_torque_rate(tau_d_calculated, tau_J_d, delta_tau_max):
    difference = tau_d_calculated - tau_J_d
    clamped_difference = torch.clamp(difference, -delta_tau_max, delta_tau_max)
    tau_d_saturated = tau_J_d + clamped_difference
    return tau_d_saturated



def euler_2_quat(euler):
    """calculates and returns: yaw, pitch, roll from given quaternion"""

    quat = R.from_euler("xyz", euler).as_quat()

    if quat[0] <0:
         quat= -1 * quat

         
    return quat
def quat_2_euler(quat):
    """calculates and returns: yaw, pitch, roll from given quaternion"""
    return R.from_quat(quat).as_euler("xyz")
