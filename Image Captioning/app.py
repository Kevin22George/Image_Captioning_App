# from image_captioning import Backend
import streamlit as st
from streamlit_option_menu import option_menu
import json
from streamlit_lottie import st_lottie
import streamlit as st
from PIL import Image, ImageDraw
import pandas as pd
import os
import pickle
import numpy as np
from googletrans import Translator
from gtts import gTTS
from playsound import playsound
from tqdm import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from nltk.translate.bleu_score import corpus_bleu

st.set_page_config(page_title = 'Image Captioning App')

st.title('Image Captioning App')
st.write('---')

ANI_DIR = 'dataset/lottiefile/'

selected = option_menu(menu_title = None,
                        options = ['Home', 'Caption Embedder', 'Caption Generator'],
                        icons=['house-fill', 'pencil-square', 'gear'],
                        menu_icon = 'cast',
                        default_index = 0,
                        orientation = 'horizontal',
                        styles={
                            'container': {'padding': '0!important',
                                          'background-color': '#313C53'},
                            'icon': {'color': 'cyan',
                                     'font-size': '25px'},
                            'nav-link': {'font-size': '19px',
                                         'text-align': 'center',
                                         'margin': '0px',
                                         '--hover-color': '#191E2A',
                                         'transition': '.3s'},                            
                            'nav-link-selected': {'background-color': '#038B74',
                                                  'font-weight': 'normal',
                                                  'cursor': 'default',
                                                  'transition': '.3s'},
                            })

if selected == 'Home':
    st.subheader('About')
    st.write('''
             Image Captioning is the process of generating textual description of an 
             image. It uses both Natural Language Processing and Computer Vision to 
             generate the captions. The dataset will be in the form [image → captions].
             The dataset consists of input images and their corresponding output captions.
             ''')
    st.write('##')
    
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f) 
        
    lottie = load_lottiefile(ANI_DIR + 'about.json')

    st_lottie(lottie, speed = 1, reverse = False, loop = True, quality = 'low', height = 350,
        width = None, key = None)
    
    st.subheader('Caption Embedder')
    st.write('Embed captions onto an image with customizable options available such as:')
    st.markdown('''<ol>
                <li style = "font-size: 17px">Selecting a caption.</li>
                <li style = "font-size: 17px">Setting the caption's alignment.</li>
                <li style = "font-size: 17px">Entering the value to display the number
                of letters for each line.</li>
                <li style = "font-size: 17px">Entering a numeric value for X-axis.</li>
                <li style = "font-size: 17px">Entering a numeric value for Y-axis.</li>
                <li style = "font-size: 17px">Picking a color for the caption.</li>
                <li style = "font-size: 17px">Download the edited image and save it to the
                specified path.</li>
                </ol>''',
                unsafe_allow_html = True)

    lottie = load_lottiefile(ANI_DIR + 'embedder.json')

    st_lottie(lottie, speed = 1, reverse = False, loop = True, quality = 'low', height = 450,
        width = None, key = None)
    
    st.subheader('Caption Generator')
    st.write('''
             A new benchmark collection for sentence-based image description and search,
              consisting of 8,000 images that are each paired with five different captions
             which provide clear descriptions of the salient entities and events. The
             images were chosen from six different Flickr groups, and tend not to contain
             any well-known people or locations, but were manually selected to depict a
             variety of scenes and situations.
             ''')
    
    st.write('''
             — Basic knowledge of two deep learning techniques, including LSTM and CNN,
             is required.
             ''')
    
    lottie = load_lottiefile(ANI_DIR + 'generator.json')

    st_lottie(lottie, speed = 1, reverse = False, loop = True, quality = 'low', height = 350,
        width = None, key = None)
    
    with st.expander('More Information'):
        st.write('Steps involved:')
        st.markdown('''<ol>
                <li style = "font-size: 17px">Import required libraries</li>
                <li style = "font-size: 17px">Extract image features using VGG16</li>
                <li style = "font-size: 17px">Load caption data</li>
                <li style = "font-size: 17px">Preprocess the caption data</li>
                <li style = "font-size: 17px">Train test split</li>
                <li style = "font-size: 17px">Create data generator function</li>
                <li style = "font-size: 17px">Model creation using CNN LSTM</li>
                <li style = "font-size: 17px">Generate captions for images</li>                    
                </ol>''',
                unsafe_allow_html = True) 

elif selected  == 'Caption Embedder':
    uploaded_file = st.file_uploader('Choose a File')

    if uploaded_file is not None:
        df = pd.read_csv('dataset\captions\captions.csv')
        # random_quote = df['Quote'].sample(n=1).values[0]
        caption = df['Quote'].tolist()
        # random.shuffle(caption)
        radio = st.radio('Choose Caption Type', ['Select Caption', 'Enter / Edit Caption'])
        
        if radio == 'Select Caption':
            caption = st.selectbox('Select Caption', caption)
            with open('type.txt', 'wb') as f:
                pickle.dump(caption, f)            

        else:
            with open("type.txt", "rb") as f:
                caption = pickle.load(f)

            caption = st.text_input('Enter / Edit Caption', caption)

        col1, col2 = st.columns([1, 1])
        alignment = col1.selectbox('Select Alignment', ['Left', 'Center', 'Right'], index = 1)
        alignment = alignment.lower()                
        letters_each_line = col2.number_input('Enter Number of Letters for Each Line', value = 1.6, step = .05, help = 'Enter a decimal number for the number of letters to be displayed in each line. Increases and decreases value by .05')
        x1 = col1.number_input('Enter X-axis', value = 412, step = 2, help = 'Increases and decreases value by 2')
        y1 = col2.number_input('Enter Y-axis', value = 412, step = 2, help = 'Increases and decreases value by 2')
        color_pick = st.color_picker('Pick a Color')

        img = Image.open(uploaded_file)
        d = ImageDraw.Draw(img)
        sum = 0

        for letter in caption:
            sum += d.textsize(letter)[0]

        avg_len_of_letter = sum / len(caption)
        no_of_letters_each_line = (x1 / letters_each_line) / avg_len_of_letter
        incrementer = 0
        fresh_sentence = ''

        for letter in caption:
            if letter == '-':
                fresh_sentence += '\n\n' + letter

            elif incrementer < no_of_letters_each_line:
                fresh_sentence += letter

            else:
                if letter == ' ':
                    fresh_sentence += '\n'
                    incrementer = 0

                else:
                    fresh_sentence += letter
                    
            incrementer += 1

        dim = d.textsize(fresh_sentence)
        x2 = dim[0]
        y2 = dim[1]
        qx = (x1/2 - x2/2)
        qy = (y1/2 - y2/2)
        d.text((qx, qy), fresh_sentence, align = alignment, fill = color_pick)

        st.image(img)
        img.save('backend/caption.png')

        # if 'image' not in st.session_state:
        #     st.session_state.image = 'not done'

        # def click_state():
        #     st.session_state.image = 'done'
        
        st.write('##')
        download = st.download_button('Download Image',
                            data = open('backend/caption.png', 'rb').read(),
                            file_name = 'image-caption.png',
                            mime = 'image/png', help = 'Download Image') # on_click = click_state
        
        if download:
            st.success('Successfully Downloaded!')

        else: pass

        # if st.session_state.image == 'done':
        #     progress_bar = st.progress(0)

            # for i in range(100):
            #     import time
            #     time.sleep(.05)
            #     progress_bar.progress(i + 1)
            # st.success('Sucessfully Downloaded!')

    else: pass

elif selected  == 'Caption Generator':
    languages = {'Arabic': 'ar', 'Chinese (simplified)': 'zh-cn',
                    'Chinese (traditional)': 'zh-tw', 'English': 'en', 'Hindi': 'hi',
                    'Japanese': 'ja', 'Kannada': 'kn', 'Korean': 'ko', 'Malayalam': 'ml',
                    'Portuguese': 'pt','Spanish': 'es'}

    lang = st.selectbox('Select Language:', languages.keys(), index = 3)
    image_name = st.file_uploader('Choose a File')    

    if image_name is not None:

        IMG_DIR = 'dataset/flickr8k_images/'
        WORKING_DIR = 'dataset/working/'
        CAPTION_DIR = 'dataset/captions/'    

        if not os.path.exists(os.path.join(WORKING_DIR, 'features.pkl')):
            model = VGG16()
            model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
            # print(model.summary())

            features = {}
            directory = os.path.join(IMG_DIR, 'Images')

            for img_name in tqdm(os.listdir(directory), desc = 'Extracting Features'):
                img_path = directory + '/' + img_name
                image = load_img(img_path, target_size=(224, 224))
                image = img_to_array(image)
                image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                image = preprocess_input(image)
                feature = model.predict(image, verbose=0)
                image_id = img_name.split('.')[0]
                features[image_id] = feature

            pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))

        else: pass

        with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
            features = pickle.load(f)

        with open(os.path.join(CAPTION_DIR, 'captions.txt'), 'r', encoding = 'utf8') as f:
            next(f)
            captions_doc = f.read()

        # mapping = {}

        # for line in tqdm(captions_doc.split('\n')):
        #     tokens = line.split(',')
            
        #     if len(line) < 2:
        #         continue

        #     image_id, caption = tokens[0], tokens[1:]
        #     image_id = image_id.split('.')[0]
        #     caption = ' '.join(caption)

        #     if image_id not in mapping:
        #         mapping[image_id] = []

        #     mapping[image_id].append(caption)

        # with open('read.txt', 'wb') as f:
        #     pickle.dump(mapping, f)

        with open("read.txt", "rb") as f:
            mapping = pickle.load(f)

        # print(len(mapping))

        def clean(mapping):
            for key, captions in mapping.items():
                for i in range(len(captions)):
                    caption = captions[i]
                    caption = caption.lower()
                    caption = caption.replace('[^A-Za-z]', '')
                    caption = caption.replace('\s+', ' ')
                    caption = 'startseq ' + ' '.join([word for word in caption.split() if len(word)>1]) + ' endseq'
                    captions[i] = caption
        clean(mapping)

        # print(mapping['1000268201_693b08cb0e'])

        # all_captions = []

        # for key in mapping:
        #     for caption in mapping[key]:
        #         all_captions.append(caption)

        # f = open('items.txt','w')
        # for item in all_captions:
        #     f.write(item + '\n')
        # f.close()

        with open('items.txt') as f:
            all_captions = f.readlines()

        # print(len(all_captions))

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_captions)
        vocab_size = len(tokenizer.word_index) + 1

        max_length = max(len(caption.split()) for caption in all_captions)
        # print(max_length)

        image_ids = list(mapping.keys())
        split = int(len(image_ids) * 0.90)
        train = image_ids[:split]
        test = image_ids[split:]

        def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
            X1, X2, y = list(), list(), list()
            n = 0
            while 1:
                for key in data_keys:
                    n += 1
                    captions = mapping[key]

                    for caption in captions:
                        seq = tokenizer.texts_to_sequences([caption])[0]

                        for i in range(1, len(seq)):
                            in_seq, out_seq = seq[:i], seq[i]
                            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]             
                            X1.append(features[key][0])
                            X2.append(in_seq)
                            y.append(out_seq)

                    if n == batch_size:
                        X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                        yield [X1, X2], y
                        X1, X2, y = list(), list(), list()
                        n = 0

        if not os.path.exists(WORKING_DIR + 'model.h5'):
            inputs1 = Input(shape=(4096,))
            fe1 = Dropout(0.4)(inputs1)
            fe2 = Dense(256, activation='relu')(fe1)
            inputs2 = Input(shape=(max_length,))
            se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
            se2 = Dropout(0.4)(se1)
            se3 = LSTM(256)(se2)

            decoder1 = add([fe2, se3])
            decoder2 = Dense(256, activation='relu')(decoder1)
            outputs = Dense(vocab_size, activation='softmax')(decoder2)

            model = Model(inputs = [inputs1, inputs2], outputs = outputs)
            model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

            epochs = 20
            batch_size = 32
            steps = len(train) // batch_size

            for i in range(epochs):
                generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
                model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)

            model.save(WORKING_DIR + 'model.h5')

        else: pass

        if not os.path.exists(WORKING_DIR + 'tokenizer.pkl'):
            with open(WORKING_DIR + 'tokenizer.pkl', 'wb') as f:
                pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

        else: pass

        model = load_model(WORKING_DIR + 'model.h5')

        with open(WORKING_DIR + 'tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)

        def idx_to_word(integer, tokenizer):
            for word, index in tokenizer.word_index.items():
                if index == integer:
                    return word
            return None

        def predict_caption(model, image, tokenizer, max_length):
            in_text = 'startseq'

            for i in range(max_length):
                sequence = tokenizer.texts_to_sequences([in_text])[0]
                sequence = pad_sequences([sequence], max_length)
                yhat = model.predict([image, sequence], verbose=0)
                yhat = np.argmax(yhat)
                word = idx_to_word(yhat, tokenizer)

                if word is None:
                    break

                in_text += ' ' + word

                if word == 'endseq':
                    break

            clean_in_text = in_text.split()
            clean_in_text = clean_in_text[1: -1]
            clean_in_text = ' '.join(clean_in_text)            
            return clean_in_text

        # actual, predicted = list(), list()

        # if not os.path.exists(WORKING_DIR + 'actual.pkl') or not os.path.exists(WORKING_DIR + 'predicted.pkl'):
        # for key in tqdm(test[:1]):
        #     captions = mapping[key]
        #     y_pred = predict_caption(model, features[key], tokenizer, max_length) 
        #     actual_captions = [caption.split() for caption in captions]
        #     y_pred = y_pred.split()
        #     actual.append(actual_captions)
        #     predicted.append(y_pred)

        # print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
        # print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))  

        # def resize_by_percentage(image_name, pct):
        #     with Image.open(image_name) as img:
        #         width, height = img.size
        #         resized_dimensions = (int(width * pct), int(height * pct))
        #         resized = img.resize(resized_dimensions)
        #         st.image(resized)

        # resize_by_percentage(image_name, 0.5)
        vgg_model = VGG16()
        vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
        image_path = image_name
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = vgg_model.predict(image, verbose=0)        
        
        st.image(image_name)

        def language(translate_to):
            translator = Translator()
            from_lang = 'en'
            to_lang = translate_to
            sentence = predict_caption(model, feature, tokenizer, 35)
            text_to_translate = translator.translate(sentence, src = from_lang,
                                                    dest = to_lang)
            text = text_to_translate.text
            row.write('Caption: '+ text.capitalize())
            speak = gTTS(text = text, lang = to_lang, slow = False)
            speak.save('translation.mp3')

        def play():
            playsound('translation.mp3')
            os.remove('translation.mp3')

        def make_uchr(code: str):
            return chr(int(code.lstrip("U+").zfill(8), 16))                                       
        
        key_list = list(languages.keys())
        val_list = list(languages.values())

        row = st.empty()
        listen = st.button(make_uchr("U+1F50A"), help = 'Listen', on_click = play)  

        for i in languages.keys():
            if lang == i:
                ind = key_list.index(i)
                language(val_list[ind])
                # if listen:
                #     play()

                # else: pass
            
            else: pass        

    else: pass

style = '''
<style>

* {font-family: sans-serif;}

header {visibility: hidden;}

a {visibility: hidden;}

footer {visibility: hidden;}

p {font-size: 20px;}

img {padding-left: 110px;}

</style>
'''
st.markdown(style, unsafe_allow_html = True)
