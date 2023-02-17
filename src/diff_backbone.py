

from .basis_funcs import *;
from .unet import *;

def temporal_gather(a: torch.Tensor, t: torch.LongTensor, x_shape):
    """Gather values from tensor `a` using indices from `t`.

    Adds dimensions at the end of the tensor to match with the number of dimensions
    of `x_shape`
    """
    batch_size = len(t)
    # print("huh",a.device,t.device)
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class Schedule:
    def __init__(self, betas):
        # Store basic information
        self.timesteps = len(betas)
        self.betas = betas
        self.alphas = 1.0 - betas

        # Pre-compute useful values:
        # use them in your code!
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.alphas_cumprod_prev = nn.functional.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )

        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )


class LinearSchedule(Schedule):
    def __init__(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        super().__init__(torch.linspace(beta_start, beta_end, timesteps))


class QuadraticSchedule(Schedule):
    def __init__(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        super().__init__(
            torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2
        )


class SigmoidSchedule(Schedule):
    def __init__(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, timesteps)
        super().__init__(torch.sigmoid(betas) * (beta_end - beta_start) + beta_start)

def q_sample(schedule: Schedule, x_0: torch.Tensor, t: torch.LongTensor, noise: Optional[torch.Tensor] = None):
    """Sample q(x_t|x_0) for a batch

    Args:
        schedule (Schedule): The $\beta_t$ schedule x_0 (torch.Tensor): A batch
        of images (N, C, W, H)

        t (torch.Tensor): A 1D tensor of integers (time)

        noise (torch.Tensor, optional): Sampled noise of the same dimension than
        x_0; if not given, sample one. Defaults to None.
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    # alphaProd = torch.prod(schedule.alphas)

    cumProd = torch.cumprod(schedule.alphas, 0)
    alphaT = temporal_gather(cumProd, t, x_0.shape)
    xt = torch.sqrt(alphaT) * x_0 + torch.sqrt(1 - alphaT) * noise
    return xt, noise



class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embeddings"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Model(nn.Module):
    def __init__(self, dim, channels=1, time_emb_dim=10, class_emb_dim=None):
        """Initialize our noise model

        Args:
            dim (int): The width/height of an image
            channels (int, optional): The number of channels of the image. Defaults to 1.
        """
        super().__init__()
        self.image_size = dim
        self.channels = channels

        # À compléter...
        # assert False, 'Code non implémenté'
        self.t_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.unet = Unet(
            # Width/Height of image
            dim,
            # Just one channel for MNIST
            channels=channels,
            # Time embeddings are in $R^d$
            time_dim=time_emb_dim,
            class_emb_dim=class_emb_dim
        )

    def noise_classes(self, batch_size):
        return -1 * torch.ones(batch_size, device=device)

    def forward(self, x_t, time, classes=None):
        # À compléter...
        time_emb = self.t_emb(time)

        # print("model",x_t.shape,time.shape,classes.shape)

        return self.unet(x_t, time_emb, classes)


@torch.no_grad()
def p_sample(schedule: Schedule, model: Model, x: torch.Tensor, t, t_index: torch.LongTensor, labels, w=0):

    postVa = schedule.posterior_variance
    input_size = x.shape

    cumu_prod = torch.cumprod(schedule.alphas, axis=0)

    alpht_time_T = temporal_gather(schedule.alphas, t, x.shape)

    alpht_time_T_bar = temporal_gather(cumu_prod, t, x.shape)

    agno_gen = model(x, t.to(device), None)
    if w > 0:
        cond_gen = model(x, t.to(device), labels)
        modOut = -w*agno_gen + (1+w)*cond_gen
    else:
        modOut = agno_gen

    sigma_t = torch.sqrt(temporal_gather(postVa, t, x.shape).to(device))

    z = torch.zeros_like(x) if t_index == 0 else torch.normal(mean=0, std=1, size=input_size)
    frac = (1 - alpht_time_T) / (torch.sqrt(1 - alpht_time_T_bar))

    prevX = (1/torch.sqrt(alpht_time_T)) * (x - frac * modOut) + sigma_t * z.to(device)

    return prevX


@torch.no_grad()
def sample(schedule, model, labels=None, batch_size=16, w=1, justLast=False):
    """Sample images from the generation model

    Args:
        schedule (Schedule): The variance schedule
        model (Model): The noise model
        batch_size (int, optional): Number of images to generate. Defaults to 16.

    Returns:
        List[torch.Tensor]: List of images for each time step $x_{T-1}, \ldots, x_0$
    """
    image_size = model.image_size
    channels = model.channels
    device = next(model.parameters()).device

    if labels is not None:
        batch_size = labels.shape[0]

    if w == 0:
        #no guidance
        labels = None;

    elif labels is None:
        #with guidance, generate one per class
        labels = torch.randint(low=0, high=9, size=(batch_size,))

    if not labels is None:
        #send to cuda
        labels = labels.to(device)

    #initialize running variable
    initSize = (batch_size,channels,image_size, image_size)
    runningX = torch.normal(mean=0, std=1, size=initSize).to(device)

    bigT = schedule.timesteps
    imgs = []
    for tIndex in range(bigT):
        rev_tIndex = bigT - 1 - tIndex
        ts = torch.Tensor([rev_tIndex] * batch_size).to(torch.int64).to(device)
        runningX = p_sample(schedule, model, runningX, ts, rev_tIndex, labels, w=w)

        if not justLast or tIndex == bigT - 1:
            imgs.append(runningX.cpu())

    return imgs


