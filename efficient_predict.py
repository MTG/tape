def efficient_predict(model, audio):
  assert model.hop_length == 256 # HARDCODED for the efficient prediction
  import torch.nn as nn
  # to device and pad
  frame = torch.tensor(audio).to(model.linear.weight.device)
  frame = nn.functional.pad(frame, pad=(model.window_size // 2, model.window_size // 2))
  # normalize and unsqueeze
  frame -= frame.mean()
  frame /= frame.std()
  frame = frame.unsqueeze(0)
  #
  model = model.eval()
  with torch.no_grad():
    attendant_hop = model.window_size//1024

    # attendant encoder require a bigger hop size, we do it with roll and shift
    attendants = model.slow(frame)
    attendants = torch.cat((attendants,
                            torch.roll(attendants,shifts=-attendant_hop, dims=2),
                            torch.roll(attendants,shifts=-2*attendant_hop, dims=2),
                            torch.roll(attendants,shifts=-3*attendant_hop, dims=2)), 
                          dim=1)
    attendants = attendants[:, :, :-(4*attendant_hop), :].permute(0, 2, 1, 3).squeeze(0)
    attendants.shape

    # main require stricter centered window
    chop_pad = model.window_size//2 - 512
    mains = model.fast(frame[:, chop_pad:-chop_pad])
    mains = torch.cat((mains,
                      torch.roll(mains,shifts=-1, dims=2),
                      torch.roll(mains,shifts=-2, dims=2),
                      torch.roll(mains,shifts=-3, dims=2)), 
                      dim=1)
    mains = mains[:, :, :-4, :].permute(0, 2, 1, 3).squeeze(0)

    # now mix the two network outputs with transformer
    attendants = model.pe(attendants)
    attendants = model.encoder1(attendants)
    attendants = model.encoder2(attendants)
    mains = model.pe(mains)
    mains = model.decoder1(mains, attendants)
    mains = model.decoder1(mains, attendants)

    # final linear layer
    salience = mains.unsqueeze(1)
    salience = salience.reshape(salience.shape[0], -1)
    salience = model.linear(salience)
    salience = torch.sigmoid(salience)
  # convert to activation, frequency, and confidence
  activation = salience.cpu()
  frequency = model.to_freq(activation, viterbi=False)
  confidence = activation.max(dim=1)[0]
  t = torch.arange(confidence.shape[0]) * model.hop_length / model.sr
  return t.numpy(), frequency, confidence.numpy(), activation.numpy()
