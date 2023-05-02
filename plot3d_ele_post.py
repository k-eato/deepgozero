#!/usr/bin/env python



import click as ck

import numpy as np

import pandas as pd

from utils import Ontology

from sklearn.manifold import TSNE

import umap


import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from deepgozero import DGELModel, load_normal_forms

import torch as th



@ck.command()

@ck.option(

    '--data-root',
'-dr', 
default='data',

    help='Prediction model')

@ck.option(

    '--ont',
'-ont',
default='mf',

    help='Prediction model')

@ck.option(

    '--model',
'-m', 
default='deepgozero',

    help='Prediction model')

def main(data_root,ont, model):

    go_file = f'{data_root}/go.norm'

    model_file = f'{data_root}/{ont}/deepgozero.th'


    # Create and load model, must match the training model

    device = 'cpu'

    go = Ontology(f'{data_root}/go.obo', with_rels=True)

    terms_dict = get_ont_terms(data_root,ont)

    ipr_df = pd.read_pickle(f'{data_root}/{ont}/interpros.pkl')

    iprs = ipr_df['interpros'].values

    iprs_dict = {v:k for k, v in  enumerate(iprs)}

    n_terms = len(terms_dict)

    n_iprs = len(iprs_dict)

    _, _, _, _,relations, zero_classes =load_normal_forms(go_file, terms_dict)

    n_rels =len(relations)

    n_zeros = len(zero_classes)

    net =DGELModel(n_iprs, n_terms, n_zeros, n_rels,device).to(device)

    net.load_state_dict(th.load(model_file,map_location=device),strict=False)



    # How to select specific GO terms

    # plot_classes = ['GO:0003674', 'GO:0005488', 'GO:0003824', ]

    # classes = [t_id for t_id in plot_classes if t_id in terms_dict]





    # Get GO embeddings

    mf_terms_dict =get_ont_terms(data_root,"mf")

    bp_terms_dict =get_ont_terms(data_root,"bp")

    cc_terms_dict =get_ont_terms(data_root,"cc")

    mf_embeds =get_embeds(mf_terms_dict,net)
    
    bp_embeds = get_embeds(bp_terms_dict,net)

    cc_embeds =get_embeds(cc_terms_dict,net)
    
    # Take a Subset
    th.manual_seed(0)
    
    mf_embeds = mf_embeds[th.randint(len(mf_embeds), (300,))]
    cc_embeds = cc_embeds[th.randint(len(cc_embeds), (300,))]
    bp_embeds = bp_embeds[th.randint(len(bp_embeds), (300,))]
    

    embeds =np.concatenate( (mf_embeds,bp_embeds,cc_embeds ),axis=0)

    labels =np.zeros(len(embeds))
    
    labels[len(mf_embeds):len(mf_embeds) + len(bp_embeds)] = 1

    labels[len(mf_embeds) + len(bp_embeds):len(mf_embeds) + len(bp_embeds) +len(cc_embeds)] = 2

    labels =labels.astype(int)
    
    print("# MF Labels = ", len(mf_embeds))
    print("# CC Labels = ", len(cc_embeds))
    print("# BP Labels = ", len(bp_embeds))
    print("# Labels = ", len(labels))
    print("# Embeds = ", len(embeds))
    
    plot_scatter(embeds,labels,model)



def get_ont_terms(data_root,ont):

    terms_df =pd.read_pickle(f'{data_root}/{ont}/terms.pkl')

    terms =terms_df['gos'].values.flatten()

    terms_dict = {v:i for i,v in enumerate(terms)}

    return terms_dict



def get_embeds(terms_dict,net):

    classes = terms_dict.keys()

    plot_ids = [terms_dict[t_id] for t_id in classes]

    embeds =net.go_embed(th.LongTensor(plot_ids)).detach().numpy()

    return embeds



def plot_embeddings(embeds,rs, classes):

    

    colors = ['b','g', 'r', 'c','m', 'y', 'k']

    embeds =TSNE().fit_transform(embeds)

    print(embeds,rs)

    fig =  plt.figure()

    ax =fig.add_subplot(111,projection='3d')

    ax.set_aspect('auto')

    

    

    for i in range(embeds.shape[0]):

        a,b = embeds[i,0], embeds[i,1]

        r =rs[i]

        u =np.linspace(0,2 * np.pi,100)

        v =np.linspace(0,np.pi,100)

        x =r * np.outer(np.cos(u),np.sin(v)) + a

        y =r * np.outer(np.sin(u),np.sin(v)) + b

        z =r * np.outer(np.ones(np.size(u)),np.cos(v))

        ax.plot_surface(x,y, z, color=colors[(i + 2) % len(colors)],rstride=4,cstride=4,linewidth=0,alpha=0.3)

        # ax.annotate(classes[i][1:-1], xy=(x, y + r + 0.03), fontsize=10, ha="center", color=colors[i % len(colors)])

    filename ='embeds3d.png'

    plt.savefig(filename)

    plt.show()



    

def plot_scatter(embeds, labels, model):

    reducer = umap.UMAP()

    embeds = reducer.fit_transform(embeds)

    # embeds = TSNE(perplexity=300).fit_transform(embeds)

    colors = np.array(['b','g', 'r'])

    plt.scatter(embeds[:,0], embeds[:, 1],c=colors[labels],marker='.',s=5)

    mf_ont =mpatches.Patch(color='b',label='MF')

    bp_ont =mpatches.Patch(color='g',label='BP')

    cc_ont =mpatches.Patch(color='r',label='CC')

    plt.legend(handles=[mf_ont,bp_ont,cc_ont])

    plt.title('UMAP Projection of GO Embeddings - El Embed PostTrain - CC ',fontsize=12)

    filename=model +"_go_embedding.png"

    plt.savefig(filename)
    
    plt.show()



if __name__ == '__main__':

    main()
