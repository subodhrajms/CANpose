def CVB(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class PNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PNorm_bimodal(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, a, **kwargs):
        x , y = a
        x_norm = self.norm(x)
        y_norm = self.norm(y)
        return self.fn((x_norm, y_norm), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FFB(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MCB(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.conv = PNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)

        
class CAB(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, w):
        x, y = w
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        qkv_n = self.to_qkv(y).chunk(3, dim=-1)
        q_n, k_n, v_n = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv_n)

        dots1 = torch.matmul(q, k_n.transpose(-1, -2)) * self.scale
        
        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots1 = dots1 + relative_bias
            
        attn1 = self.attend(dots1)
        out1 = torch.matmul(attn1, v_n)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out1 = self.to_out(out1)
        dots2 = torch.matmul(q_n, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots2 = dots2 + relative_bias

        attn2 = self.attend(dots2)
        out2 = torch.matmul(attn2, v)
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out2 = self.to_out(out2)
        return out1, out2

class CustomSequential(nn.Module):
    def __init__(self, module_list):
        super(CustomSequential, self).__init__()
        self.module_list = nn.ModuleList(module_list)

    def forward(self, inputs):
        # Unpack the tuple
        input1, input2 = inputs
        # Apply the first rearrange operation separately to each input
        output1 = self.module_list[0](input1)
        output2 = self.module_list[0](input2)

        # Make a tuple out of the results
        output_tuple = (output1, output2)
        # Apply the remaining modules in the module list sequentially
        for module in self.module_list[1:2]:
            output_tuple_1,output_tuple_2 = module(output_tuple)
        for module in self.module_list[2:]:
            output1 = module(output_tuple_1)
            output2 = module(output_tuple_2)
        output_tuple = (output1, output2)

        return output_tuple

class LoGo(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)
        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = CAB(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FFB(oup, hidden_dim, dropout)

        self.attn = CustomSequential([
            Rearrange('b c ih iw -> b (ih iw) c'),  # First rearrange for the first input
            PNorm_bimodal(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
            ])

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, a):
        x,y = a
        if self.downsample:
            x_pool2 = self.pool2(x)
            y_pool2 = self.pool2(y)
            p,q = self.attn((x_pool2,y_pool2))
            x = self.proj(self.pool1(x)) + p
            y = self.proj(self.pool1(y)) + q
        else:
            p,q = self.attn((x,y))
            x = x + p
            y = y + q
            
        x = x + self.ff(x)
        y = y + self.ff(y)
        return x,y        

class CANpose(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes, block_types=['M', 'L', 'M', 'L']):
        super().__init__()
        ih, iw = image_size
        block = {'M': MCB, 'L': LoGo}

        self.s0 = self._make_layer(
            CVB, in_channels, channels[0], num_blocks[0], (ih//2 , iw//2))
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih//4 , iw//4))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih//8, iw//8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih//16, iw//16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih//32, iw//32))

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x,y):
        x = self.s0(x)
        y = self.s0(y)
        x = self.s1(x)
        y = self.s1(y)
        x,y = self.s2((x,y))
        x = self.s3(x)
        y = self.s3(y)
        x,y = self.s4((x,y))

        x = self.pool(x).view(-1, x.shape[1])
        y = self.pool(y).view(-1, y.shape[1])
        z = torch.add(x,y)
        z = self.fc(z)
        return z


    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)


def canpose(Input_shape = None,number_classes=None):
    num_blocks = [2, 2, 2, 2, 2]            # L
    channels = [64, 96, 192, 384, 768]      # D
    return CANpose(Input_shape, 3, num_blocks, channels, num_classes=number_classes)
