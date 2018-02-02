import importlib, utils2; 
importlib.reload(utils2)
from utils2 import *
import sys

np.set_printoptions(4)

cfg = K.tf.ConfigProto(gpu_options = {'allow_growth': True})
K.set_session(K.tf.Session(config = cfg))

def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip() ]

def parse_stories(lines):
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid,line = line.split(" ",1)
        if int(nid) == 1:story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            substory = [[str(i)+":"]+x for i,x in enumerate(story) if x]
            data.append((substory, q, a))
            story.append('')
        else: story.append(tokenize(line))
    return data

path = get_file('babi-tasks-v1-2.tar.gz', origin = 'https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')

tar = tarfile.open(path)

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
    'two_supporting_facts_1k': 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt',
}

challenge_type = 'single_supporting_fact_10k'

challenge = challenges[challenge_type]

def get_stories(file):
    data = parse_stories(file.readlines())
    return [(story, question, answer) for story, question, answer in data]

train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))

stories = train_stories + test_stories

story_maxlen =    max((len(x) for s,_,_ in stories for x in s))
story_maxsents = max((len(x) for x, _, _ in stories))
query_maxlen = max(len(x) for _, x, _ in stories)

def create_vocab(stories):
    vocab = set()
    for i,story in enumerate(stories):
        sys.stdout.write("\r Running story number: " + str(i))
        
        #Getting vocab from stories
        for text in story[0]:
            [vocab.add(word) for word in text ]
        sys.stdout.flush()
        
        #getting vocab from questions
        [vocab.add(word) for word in story[1] ]
        
        #Getting vocab from Answer
        vocab.add(story[2])
    return vocab

vocab = sorted(create_vocab(stories))
vocab.insert(0, '<PAD>')
vocab_size = len(vocab)

word_idx = dict((c, i) for i, c in enumerate(vocab))

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []; Xq = []; Y = []
    for story, query, answer in data:
        x = [[word_idx[w] for w in s] for s in story]
        xq = [word_idx[w] for w in query]
        y = [word_idx[answer]]
        X.append(x); Xq.append(xq); Y.append(y)
    return ([pad_sequences(x, maxlen=story_maxlen) for x in X],
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))


inputs_train, queries_train, answers_train = vectorize_stories(train_stories, 
     word_idx, story_maxlen, query_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories, 
     word_idx, story_maxlen, query_maxlen)


def stack_inputs(inputs):
    for i,it in enumerate(inputs):
        inputs[i] = np.concatenate([it, 
                           np.zeros((story_maxsents-it.shape[0],story_maxlen), 'int')])
    return np.stack(inputs)
inputs_train = stack_inputs(inputs_train)
inputs_test = stack_inputs(inputs_test)


inps = [inputs_train, queries_train]
val_inps = [inputs_test, queries_test]


emb_dim = 20
parms = {'verbose': 2}

def emb_sent_bow(inp):
    emb = TimeDistributed(Embedding(vocab_size, emb_dim))(inp)
    return Lambda(lambda x: K.sum(x, 2))(emb)
	
inp_story = Input((story_maxsents, story_maxlen))
emb_story = emb_sent_bow(inp_story)
inp_story.get_shape(), emb_story.get_shape()

inp_q = Input((query_maxlen,))

emb_q = Embedding(vocab_size, emb_dim)(inp_q)
emb_q = Lambda(lambda x: K.sum(x, 1))(emb_q)
emb_q = Reshape((1, emb_dim))(emb_q)
inp_q.get_shape(), emb_q.get_shape()


x = merge([emb_story, emb_q], mode='dot', dot_axes=2)
x = Reshape((story_maxsents,))(x)
x = Activation('softmax')(x)
match = Reshape((story_maxsents,1))(x)

emb_c = emb_sent_bow(inp_story)
x = merge([match, emb_c], mode='dot', dot_axes=1)
response = Reshape((emb_dim,))(x)
res = Dense(vocab_size, activation='softmax')(response)


answer = Model([inp_story, inp_q], res)

answer.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

K.set_value(answer.optimizer.lr, 1e-2)
hist=answer.fit(inps, answers_train, **parms, nb_epoch=4, batch_size=32,
           validation_data=(val_inps, answers_test))
		   
f = Model([inp_story, inp_q], match)

qnum=512

l_st = len(train_stories[qnum][0])+1
print(train_stories[qnum])
np.squeeze(f.predict([inputs_train[qnum:qnum+1], queries_train[qnum:qnum+1]]))[:l_st]
answers_train[qnum:qnum+10,0]
answer = np.argmax(answer.predict([inputs_train[qnum:qnum+10], queries_train[qnum:qnum+10]]), 1)
from collections import Counter
data = Counter(answer)
print("The answer is :", vocab[answers_train[qnum][0]])