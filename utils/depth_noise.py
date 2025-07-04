import torch
import torch.nn.functional as F

class DepthNoise(torch.nn.Module):
    def __init__(self, 
        focal_length, 
        baseline,
        min_depth=0.1,
        max_depth=10.0,
        filter_size=3,
        inlier_thred_range=(0.01, 0.05), 
        prob_range=(0.5, 0.8),
        invalid_disp=1e7
    ):
        """
        A Simply PyTorch module to add realistic noise to depth images.
        
        Args:
            focal_length (float): Focal length of the camera (in pixels).
            baseline (float): Baseline distance between stereo cameras (in meters).
            min_depth (float): Minimum depth value after clamping.
            max_depth (float): Maximum depth value after clamping.
            filter_size (int): Kernel size for local mean disparity computation. (tuning based on image resolution)
            inlier_threshold_range (tuple): Threshold range for disparity difference, (larger values will be make less noise).
            prob_range (tuple): Probability range for matching pixels.
            invalid_disp (float): Invalid disparity
        
        """
        super().__init__()
        self.focal_length = focal_length
        self.baseline = baseline
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.invalid_disp = invalid_disp
        self.inlier_thred_range = inlier_thred_range
        self.prob_range = prob_range
        self.filter_size = filter_size
        
        weights, substitutes = self._compute_weights(filter_size)
        self.register_buffer('weights', weights.view(1, 1, filter_size, filter_size))
        self.register_buffer('substitutes', substitutes.view(1, 1, filter_size, filter_size))
        
        
    def _compute_weights(self, filter_size):
        """
        Compute weights and substitutes for disparity filtering.

        Args:
            filter_size (int): Kernel size for local mean disparity computation.
        """
        center = filter_size // 2
        idx = torch.arange(filter_size) - center
        x_filter, y_filter = torch.meshgrid(idx, idx, indexing='ij')
        sqr_radius = x_filter ** 2 + y_filter ** 2
        sqrt_radius = torch.sqrt(sqr_radius)
        weights = 1 / torch.where(sqr_radius == 0, torch.ones_like(sqrt_radius), sqrt_radius)
        weights = weights / weights.sum()
        fill_weights = 1 / (1 + sqrt_radius)
        fill_weights = torch.where(sqr_radius > filter_size, -1.0, fill_weights)
        substitutes = (fill_weights > 0).float()
        
        return weights, substitutes

    def filter_disparity(self, disparity):
        """
        Filter the disparity map using local mean disparity.
        
        Args:
            disparity (torch.Tensor): Input disparity map tensor of shape (B, C, H, W).
        """
        B, _, H, W = disparity.shape
        device = disparity.device
        center = self.filter_size // 2

        output_disparity = torch.full_like(disparity, self.invalid_disp)

        prob = torch.rand(B, 1, 1, 1, device=device) * (self.prob_range[1] - self.prob_range[0]) + self.prob_range[0]
        random_mask = (torch.rand(B, 1, H, W, device=device) < prob)

        # Compute mean disparity
        weighted_disparity = F.conv2d(disparity, self.weights, padding=center)

        # Compute differences
        differences = torch.abs(disparity - weighted_disparity)

        # Create update mask
        threshold = torch.rand(B, 1, 1, 1, device=device) * (self.inlier_thred_range[1] - self.inlier_thred_range[0]) + self.inlier_thred_range[0]
        update_mask = (differences < threshold) & random_mask

        # Compute output value: round with 1/8 precision
        output_value = torch.round(disparity * 8.0) / 8.0

        # Update output disparity
        output_disparity = torch.where(update_mask, output_value, output_disparity)

        # Apply substitutes to fill neighboring pixels
        filled_values = F.conv2d(update_mask.float() * output_value, self.substitutes, padding=center)
        counts = F.conv2d(update_mask.float(), self.substitutes, padding=center) + 1e-9
        average_filled_values = filled_values / counts
        output_disparity = torch.where(counts >= 1, average_filled_values, output_disparity)

        return output_disparity

    def forward(self, depth, add_noise: bool) -> torch.Tensor:
        # correct input shape

        if len(depth.shape) == 3:
            depth = depth.unsqueeze(1) # add channel dimension
        
        # check dimension (B, 1, H, W)
        assert depth.shape[1] == 1, "Input depth tensor must have shape (B, 1, H, W)."
        assert len(depth.shape) == 4, "Input depth tensor must have shape (B, 1, H, W)."
        
        # Clamp the depth values
        depth = torch.clamp(depth, min=self.min_depth, max=self.max_depth)
        
        if add_noise:
            # Step 1: Convert depth to disparity
            disparity = self.focal_length * self.baseline / depth

            # Step 2: Filter the disparity map
            filtered_disparity = self.filter_disparity(disparity)

            # Step 3: Recompute depth from disparity
            depth = self.focal_length * self.baseline / filtered_disparity
            depth[filtered_disparity == self.invalid_disp] = 0

            # Step 4: Clamp the depth values
            depth = torch.clamp(depth, min=0.0, max=self.max_depth) # 0 means invalid depth, e.g. nan

        return depth
