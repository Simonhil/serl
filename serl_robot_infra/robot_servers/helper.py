import torch










def pseudo_inverse(M : torch.Tensor, damped=True):
    lambda_ = 0.2 if damped else 0.0

    # Perform SVD
    U, s, Vt = torch.svd(M, full_matrices=False)
    
    # Create the diagonal matrix S
    S = torch.zeros_like(M.T)
    
    # Compute the pseudoinverse of the singular values
    for i in range(len(s)):
        S[i, i] = s[i] / (s[i]**2 + lambda_**2)
    
    # Compute the pseudoinverse
    M_pinv = Vt.T @ S.T @ U.T
    
    return M_pinv

def saturate_torque_rate(tau_d_calculated, tau_J_d, delta_tau_max):
    difference = tau_d_calculated - tau_J_d
    clamped_difference = torch.clamp(difference, -delta_tau_max, delta_tau_max)
    tau_d_saturated = tau_J_d + clamped_difference
    return tau_d_saturated