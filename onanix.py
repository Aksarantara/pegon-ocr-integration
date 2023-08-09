import os
import torch
import pickle

from jawi_models import Encoder, Decoder, Seq2Seq


device = "cpu"
jawi_dim = 58
rumi_dim = 36
enc_emb_dim = 300
dec_emb_dim = 300
enc_hidden_dim = 512
dec_hidden_dim = 512
rnn_type = "lstm"
enc2dec_hid = True
attention = True
enc_layers = 1
dec_layers = 2
m_dropout = 0
enc_bidirect = True
enc_outstate_dim = enc_hidden_dim * (2 if enc_bidirect else 1)


enc_jawi = Encoder(
    input_dim=jawi_dim,
    embed_dim=enc_emb_dim,
    hidden_dim=enc_hidden_dim,
    rnn_type=rnn_type,
    layers=enc_layers,
    dropout=m_dropout,
    device=device,
    bidirectional=enc_bidirect,
)
dec_jawi = Decoder(
    output_dim=jawi_dim,
    embed_dim=dec_emb_dim,
    hidden_dim=dec_hidden_dim,
    rnn_type=rnn_type,
    layers=dec_layers,
    dropout=m_dropout,
    use_attention=attention,
    enc_outstate_dim=enc_outstate_dim,
    device=device,
)
enc_rumi = Encoder(
    input_dim=rumi_dim,
    embed_dim=enc_emb_dim,
    hidden_dim=enc_hidden_dim,
    rnn_type=rnn_type,
    layers=enc_layers,
    dropout=m_dropout,
    device=device,
    bidirectional=enc_bidirect,
)
dec_rumi = Decoder(
    output_dim=rumi_dim,
    embed_dim=dec_emb_dim,
    hidden_dim=dec_hidden_dim,
    rnn_type=rnn_type,
    layers=dec_layers,
    dropout=m_dropout,
    use_attention=attention,
    enc_outstate_dim=enc_outstate_dim,
    device=device,
)

model_r2j = Seq2Seq(enc_rumi, dec_jawi, pass_enc2dec_hid=enc2dec_hid, device=device)
model_j2r = Seq2Seq(enc_jawi, dec_rumi, pass_enc2dec_hid=enc2dec_hid, device=device)

r2j_weights_path = os.path.join(
    "Transliterate", "r2j_10_epoch", "Training_1", "weights", "Training_1_model.pth"
)
j2r_weights_path = os.path.join(
    "Transliterate", "j2r_30_epoch", "Training_2", "weights", "Training_2_model.pth"
)

state_dict_r2j = torch.load(
    r2j_weights_path,
    map_location=torch.device("cpu"),
)
state_dict_j2r = torch.load(
    j2r_weights_path,
    map_location=torch.device("cpu"),
)

model_r2j.load_state_dict(state_dict_r2j)
model_j2r.load_state_dict(state_dict_j2r)

# src = torch.randint(0, 36, size=(512, 43), dtype=torch.int32)
# tgt = torch.randint(0, 36, size=(512, 44), dtype=torch.int32)
# src_sz = torch.randn(512)

with open("src_ex.pkl", "rb") as file:
    src = pickle.load(file)

with open("src_sz_ex.pkl", "rb") as file:
    src_sz = pickle.load(file)

with open("tgt_ex.pkl", "rb") as file:
    tgt = pickle.load(file)

args_j2r = {
    "src": src,
    "tgt": tgt,
    "src_sz": src_sz
}

args_r2j = {
    "src": tgt,
    "tgt": src,
    "src_sz": src_sz
}

torch.onnx.export(model_r2j, args_r2j, "r2j.onnx", verbose=True)
torch.onnx.export(model_j2r, args_j2r, "j2r.onnx", verbose=True)
