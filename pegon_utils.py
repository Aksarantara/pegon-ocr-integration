import json
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image
from collections import OrderedDict

PEGON_CHARS = ['_'] + [' '] + list(OrderedDict.fromkeys('َ ِ ُ ً ٍ ٌ ْ ّ َ ٰ ࣤ \u06e4 \u0653 ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩ ٠ ة ح چ ج ث ت ب ا أ إ آ ؤ ى س ز ر ࢮ ڎ ذ د خ ع ظ ڟ ط ض ص ش ڮ ك ق ڤ ف ڠ غ ي ه و ۑ ئ ن م ل ۔ : ؛ ، ( ) ! ؟ « » ۞ ء'.split())) + ['\ufffd']
CHAR_MAP = {letter: idx for idx, letter in enumerate(PEGON_CHARS)}

PEGON_CHARS_V2 = [
    '_',
    ' ',
    '"',
    "'",
    '(',
    ')',
    ':',
    '«',
    '»',
    '،',
    '؛',
    '؟',
    'ء',
    'آ',
    'أ',
    'ؤ',
    'إ',
    'ئ',
    'ا',
    'ب',
    'ة',
    'ت',
    'ث',
    'ج',
    'ح',
    'خ',
    'د',
    'ذ',
    'ر',
    'ز',
    'س',
    'ش',
    'ص',
    'ض',
    'ط',
    'ظ',
    'ع',
    'غ',
    'ػ',
    'ؼ',
    'ؽ',
    'ؾ',
    'ؿ',
    'ـ',
    'ف',
    'ق',
    'ك',
    'ل',
    'م',
    'ن',
    'ه',
    'و',
    'ى',
    'ي',
    'ً',
    'ٌ',
    'ٍ',
    'َ',
    'ُ',
    'ِ',
    'ّ',
    'ْ',
    'ٓ',
    'ٔ',
    'ٕ',
    'ٖ',
    '٠',
    '١',
    '٢',
    '٣',
    '٤',
    '٥',
    '٦',
    '٧',
    '٨',
    '٩',
    '٬',
    '٭',
    'ٮ',
    'ٰ',
    'ٱ',
    'ٴ',
    'ٶ',
    'پ',
    'څ',
    'چ',
    'ڊ',
    'ڎ',
    'ڟ',
    'ڠ',
    'ڤ',
    'ڨ',
    'ک',
    'ڬ',
    'ڮ',
    'گ',
    'ڳ',
    'ڽ',
    'ۋ',
    'ی',
    'ۏ',
    'ۑ',
    '۔',
    'ݘ',
    'ݢ',
    'ࢨ',
    'ࢮ',
    'ࣤ',
    '\ufffd'
]

CHAR_MAP_V2 = {letter: idx for idx, letter in enumerate(PEGON_CHARS_V2)}

# label transforms

normalization_table = [
    ('ٙ', 'ٓ'), # zwarakay
    ('ٞ', 'َ'), # fatha with two dots
    ('ٟ', 'ٕ'), # wavy hamza below
    ('ڪ', 'ك'), # swash kaf
    ('ۃ', 'ة'),  # ta marbuta akhir
    
    ('ࣰ','ً'),
    
    ('ࣱ','ٌ'),
    
    ('ࣲ','ٍ'),
    
    ('ﮬ', 'ه'),
    
    ('ﷲ', 'اللّٰه'),
    ('‘', '\''),
    ('’', '\''),
    ('“', '"'),
    ('”', '"'),
    ('ﺃ', 'أ'),
    ('ﺎ','ا'),
    ('ﺑ','ب'),
    ('ﺓ','ة'),
    ('ﺗ','ت'),
    ('ﺘ','ت'),
    ('ﺪ','د'),
    ('ﺮ','ر'),
    ('ﺴ','س'),
    ('ﺷ','ش'),
    ('ﺻ','ص'),
    ('ﺿ','ض'),
    ('ﻋ','ع'),
    ('ﻖ','ق'),
    ('ﻙ','ك'),
    ('ﻛ','ك'),
    ('ﻞ','ل'),
    ('ﻟ','ل'),
    ('ﻠ','ل'),
    ('ﻢ','م'),
    ('ﻣ','م'),
    ('ﻥ','ن'),
    ('ﻨ','ن'),
    ('ﻬ','ه'),
    ('ﻭ','و'),
    ('ﻮ','و'),
    ('ﻰ','ى'),
    ('ﻲ','ي'),
    ('ﻷ','لأ'),
    ('ﻻ','لا'),
    ('ﻼ','لا'),
    ('ﭪ', 'ڤ'),
    ('ٗ','ُ'),
    ('ٝ','ُ'),
    (',', '،'),
    (';', '؛'),
    ('٫','،'),
    ('۰','٠'),
    ('۱','١'),
    ('۲','٢'),
    ('۳','٣'),
    ('۴','٤'),
    ('۵','٥'),
    ('۶','٦'),
    ('۷','٧'),
    ('۸','٨'),
    ('۹','٩'),
]

def unicode_escape(label):
    return ''.join(filter(lambda c:unicodedata.category(c)[0] != 'C', label))

def arabic_normalize(label):
    for target, sub in normalization_table:
        label = re.sub(target, sub, label)
    return label

def filename_to_label(filename):
    return filename

import unicodedata, re
from torchvision.transforms import transforms
from PIL import Image

def ctc_variable_size_collate_fn(batch):
    
    batch = sorted(batch, key=lambda x: x[0].shape[2], reverse=True)
    
    images = [item[0] for item in batch]
    labels = [torch.Tensor(item[1]) for item in batch]
    label_lengths = torch.LongTensor([len(label) for label in labels])
    
    labels = torch.cat((labels))

    return images, labels, label_lengths


def ctc_collate_fn(batch):
    
    batch = sorted(batch, key=lambda x: x[0].shape[2], reverse=True)
    
    images = [item[0] for item in batch]
    labels = [torch.Tensor(item[1]) for item in batch]
    label_lengths = torch.LongTensor([len(label) for label in labels])
    
    labels = torch.cat((labels))

    images = torch.stack(images, dim=0)

    return images, labels, label_lengths

class CTCDecoder:
    def __init__(self, model, char_map, blank_char='-'):
        self.model = model.eval()
        self.alphabet = list(char_map.keys())
        self.blank_char = blank_char
        blank_idxs = [idx for idx, x in enumerate(self.alphabet) if x == self.blank_char]
        assert len(blank_idxs) == 1
        self.blank_index = blank_idxs[0]
   
    @classmethod
    def from_path(cls, weight_path, model_path, model_class,
                  char_map, blank_char):
        saved_model = model_class(**json.load(open(model_path)))
        saved_model.load_state_dict(torch.load(weight_path))
        return cls(model=saved_model,
                   char_map=char_map,
                   blank_char=blank_char)

    def convert_to_text(self, output):
        output = torch.argmax(output, dim=2).detach().cpu().numpy()
        texts = []
        for i in range(output.shape[0]):
            text = ''
            for j in range(output.shape[1]):
                if output[i, j] != self.blank_index and (j == 0 or output[i, j] != output[i, j-1]):
                    text += self.alphabet[output[i, j]]
            texts.append(text)
        return texts
    
    def infer(self, data):
        model_out = self.model(data)
        return self.convert_to_text(model_out)

class BestPathDecoder(CTCDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_chars = [x for x in self.alphabet if x != self.blank_char]
        
    def convert_to_text(self, output):
        output = torch.roll(F.softmax(output, dim=2), -1, 2).detach().cpu().numpy()
        texts = []
        for i in range(output.shape[0]):
            texts.append(best_path(output[i, :, :], self.all_chars))
        return texts
