import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import owlready2 as owl
from owlready2 import *
owlready2.reasoning.JAVA_MEMORY = 200000

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from create_geometric_interpretation import SCALE_FACTOR, GeometricInterpretation

torch.manual_seed(33)

class FaithEL(nn.Module):
    def __init__(self, emb_dim, phi, radius, gamma, psi,
                 individual_vocabulary,
                 concept_vocabulary,
                 role_vocabulary,
                 ):
        
        super(FaithEL, self).__init__()
        self.emb_dim = emb_dim
        self.phi = phi # Weights how much the concept/role parameter moves
        self.gamma = gamma # Weights how much the individual embeddings moves
        self.psi = psi
        self.radius = radius

        self.individual_embedding_dict = nn.Embedding(len(individual_vocabulary),
                                                      emb_dim
                                                      )
        with torch.no_grad():
            value = SCALE_FACTOR/2
            std_dev = 0.7
            self.individual_embedding_dict.weight.data.normal_(mean=value, std=std_dev)

        # Initializes the moving concept parameters
        
        self.concept_embedding_dict = nn.Embedding(len(concept_vocabulary),
                                                   emb_dim
                                                   )
        
        with torch.no_grad():
            for concept_name, concept_idx in concept_vocabulary.items():
                self.concept_embedding_dict.weight[concept_idx] = torch.tensor(GeometricInterpretation.concept_geointerps_dict[concept_name].centroid)

        self.role_head_embedding_dict = nn.Embedding(len(role_vocabulary),
                                                emb_dim
                                                )
        
        # Initializes the moving parameter for roles at the role's respective centroid
        with torch.no_grad():
            for role_name, role_idx in role_vocabulary.items():
                self.role_head_embedding_dict.weight[role_idx] = torch.tensor(GeometricInterpretation.role_geointerps_dict[role_name].centroid[:self.emb_dim])

        self.role_tail_embedding_dict = nn.Embedding(len(role_vocabulary),
                                                     emb_dim
                                                     )
                                                     
        with torch.no_grad():
            for role_name, role_idx in role_vocabulary.items():
                self.role_tail_embedding_dict.weight[role_idx] = torch.tensor(GeometricInterpretation.role_geointerps_dict[role_name].centroid[self.emb_dim:])

    def forward(self, data):
    
        # Concept assertions are of the form ['Concept', 'Entity']
        # Role assertions are of the form ['SubjectEntity', 'Role', 'ObjectEntity']
        
        subj_entity_idx = 1 if len(data[0]) == 2 else 0 # Checks whether the model has received a C assert or R assert

        if subj_entity_idx == 1:

            concept_idx = 0
            concept = data[:, concept_idx]
            subj_entity = data[:, subj_entity_idx]

            neg_object_entity = torch.randint(0, self.individual_embedding_dict.weight.shape[0], (subj_entity.shape))
            #neg_object_entity = self.negative_sampler_concept(data)

            out1 = self.concept_embedding_dict(concept) # Concept parameter
            out2 = self.individual_embedding_dict(subj_entity) # Subject entity parameter
            out3 = self.individual_embedding_dict(neg_object_entity) # Negatively sampled parameter
            out4 = None # Only a placeholder, has no bearing in calculating anything.

            return out1, out2, out3, out4

        elif subj_entity_idx == 0:

            role_idx = 1
            obj_entity_idx = 2
            
            role = data[:, role_idx]

            subject_entity = data[:, subj_entity_idx]
            object_entity = data[:, obj_entity_idx]
            neg_object_entity = torch.randint(0, self.individual_embedding_dict.weight.shape[0], (subject_entity.shape))
            #neg_object_entity = self.negative_sampler_role(data)

            out1 = self.role_head_embedding_dict(role) # Role parameter embedding
            out1_2 = self.role_tail_embedding_dict(role)
            out2 = self.individual_embedding_dict(subject_entity)
            out3 = self.individual_embedding_dict(object_entity)
            out4 = self.individual_embedding_dict(neg_object_entity)

            return out1, out1_2, out2, out3, out4
        
    def concept_parameter_constraint(self):
        with torch.no_grad():
            for idx, weight in enumerate(self.concept_embedding_dict.weight):
                centroid = torch.tensor(GeometricInterpretation.concept_geointerps_dict[list(GeometricInterpretation.concept_geointerps_dict.keys())[idx]].centroid)
                distance = torch.dist(weight, centroid, p=2)
                if distance > self.radius:
                    self.concept_embedding_dict.weight[idx] = centroid + self.radius * (weight - centroid) / distance

    def role_parameter_constraint(self):
        with torch.no_grad():
            for idx, head_weight in enumerate(self.role_head_embedding_dict.weight):
                centroid = torch.tensor(GeometricInterpretation.role_geointerps_dict[list(GeometricInterpretation.role_geointerps_dict.keys())[idx]].centroid)
                tail_weight = self.role_tail_embedding_dict.weight[idx]
                head_tail_concat = torch.cat((head_weight, tail_weight), dim=0)
                distance = torch.dist(head_tail_concat, centroid, p=2)
                if distance > self.radius:
                    concat_clipped_distance = centroid + self.radius * (head_tail_concat - centroid) / distance
                    self.role_head_embedding_dict.weight[idx] = concat_clipped_distance[:self.emb_dim]
                    self.role_tail_embedding_dict.weight[idx] = concat_clipped_distance[self.emb_dim:]

    # Sorts the batch row-wise e.g. unsorted = [[2,1], [0,3], [0,1]], sorted = [[0,1], [0,3], [2,1]]
    def concept_batch_sorter(self, data):
        sorted_databatch = data[data[:, 1].argsort (dim = 0, stable = False)]
        sorted_databatch = sorted_databatch[sorted_databatch[:, 0].argsort (dim = 0, stable = True)]

        return sorted_databatch
    
    def negative_sampler_concept(self, data):

        with torch.no_grad():

            sorted_databatch = self.concept_batch_sorter(data)
            
            neg_candidates = torch.randint(0, self.individual_embedding_dict.weight.shape[0], (sorted_databatch.shape[0], 1))
            corrupted_databatch = torch.cat((sorted_databatch[:,0].unsqueeze(1), neg_candidates), dim=1)
            sorted_corrupted_databatch = self.concept_batch_sorter(corrupted_databatch)

            counter = 0
            MAX_ITER = 100 # This is necessary to avoid non-terminating loops.

            negsamp_checker = sorted_databatch[:,1] == sorted_corrupted_databatch[:,1]

            while torch.any(negsamp_checker) and counter != MAX_ITER:
                conflict_idcs = torch.where(negsamp_checker == True)[0]
                for idx in conflict_idcs:
                    sorted_corrupted_databatch[idx][1] = torch.randint(0, self.individual_embedding_dict.weight.shape[0], (1,)).item()

                negsamp_checker = sorted_databatch[:,1] == sorted_corrupted_databatch[:,1]
                counter += 1

                if counter == MAX_ITER:
                    print('================Neg sampling CONCEPT loop limit reached====================')

            return sorted_corrupted_databatch[:,1]
    
    def role_batch_sorter(self, data, head_tail_result):
        
        # Handles the sorting for (h, r, ?) type queries
        if head_tail_result:
            sorted_databatch = data[data[:, 2].argsort (dim=0, stable=False)]
            sorted_databatch = sorted_databatch[sorted_databatch[:,1].argsort (dim=0, stable=True)]
            sorted_databatch = sorted_databatch[sorted_databatch[:,0].argsort (dim=0, stable=True)]
        
        # Handles the sorting for (?, r, t) type queries
        else:
            sorted_databatch = data[data[:, 2].argsort (dim=0, stable=False)]
            sorted_databatch = sorted_databatch[sorted_databatch[:,1].argsort (dim=0, stable=True)]
            sorted_databatch = sorted_databatch[sorted_databatch[:,0].argsort (dim=0, stable=True)]

        return sorted_databatch

    def negative_sampler_role(self, data):

        with torch.no_grad():
            
            head_or_tail = torch.randint(0, 2, (1,)) # If head_or_tail == True, corrupt the tail of the triple
            
            '''HARDCODED COIN TOSS RESULT'''
            head_or_tail = 1 
            corrupt_idx = 2 if head_or_tail else 0

            sorted_databatch = self.role_batch_sorter(data, head_or_tail)

            neg_candidates = torch.randint(0, self.individual_embedding_dict.weight.shape[0], (sorted_databatch.shape[0], 1))

            if head_or_tail == True:
                corrupted_databatch = torch.cat((sorted_databatch[:, :2], neg_candidates), dim=1)
                sorted_corrupted_databatch = self.role_batch_sorter(corrupted_databatch, head_or_tail)
            else:
                corrupted_databatch = torch.cat((neg_candidates, sorted_databatch[:, 1:]), dim=1)
                sorted_corrupted_databatch = self.role_batch_sorter(corrupted_databatch, head_or_tail)

            counter = 0
            MAX_ITER = 5 # This is necessary to avoid non-terminating loops.
            
            if head_or_tail == True:
                negsamp_checker = sorted_databatch[:,corrupt_idx] == sorted_corrupted_databatch[:,corrupt_idx]
            else:
                negsamp_checker = sorted_databatch[:,corrupt_idx] == sorted_corrupted_databatch[:,corrupt_idx]

            while torch.any(negsamp_checker) and counter != MAX_ITER:
                conflict_idcs = torch.where(negsamp_checker == True)[0]

                for idx in conflict_idcs:
                    sorted_corrupted_databatch[idx][corrupt_idx] = torch.randint(0, self.individual_embedding_dict.weight.shape[0], (1,)).item()

                negsamp_checker = sorted_databatch[:,corrupt_idx] == sorted_corrupted_databatch[:,corrupt_idx]
                counter += 1

                if counter == MAX_ITER:
                    print('================Neg sampling ROLE loop limit reached====================')
                    
        return sorted_corrupted_databatch[:,corrupt_idx]

        

