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
import random

from create_canonical_model import RESTRICT_LANGUAGE, INCLUDE_TOP, canmodel, CanonicalModel # Creates the canonical model and stores it in the 'canmodel' variable

SCALE_FACTOR = 1 # Controls the one-hot encoding factor for the embedding function. It means the "one-hot" dimensions will have the chosen value.
INCLUDE_TOP = INCLUDE_TOP

'''
Utility functions for initializing
the class EntityEmbedding. They
allow us to access dictionaries
containing indexes and canonical
interpretation of concepts
and roles as class.
'''

def get_concept_names_idx_dict(canmodel, include_top_flag):
   
   conceptnames_idx_dict = {concept_name: idx for idx, concept_name in enumerate(canmodel.concept_canonical_interpretation.keys())}

   if include_top_flag == True:
       conceptnames_idx_dict = {concept_name: idx-1 for idx, concept_name in enumerate(canmodel.concept_canonical_interpretation.keys())}
       conceptnames_idx_dict.pop('Thing')
       
   return conceptnames_idx_dict

def get_role_names_idx_dict(canmodel):
    rolenames_idx_dict = {role_name: idx for idx, role_name in enumerate(canmodel.role_canonical_interpretation.keys())}
    return rolenames_idx_dict

def get_entities_idx_dict(canmodel):
    entities_idx_dict = {entity: idx for idx, entity in enumerate(canmodel.domain.keys())}
    return entities_idx_dict

def get_domain_dict(canmodel):
    return canmodel.domain

''' 
Class for obtaining the positional 
embedding for each entity in the domain
of the canonical interpretation.
It represents the Mu Function from the
paper.
'''

class EntityEmbedding:

    # Dictionaries for storing the indices of concept names and role names, entities pairs, respectively
    # Keys are strings and values are integers
    
    concept_names_idx_dict = get_concept_names_idx_dict(canmodel, INCLUDE_TOP)
    role_names_idx_dict = get_role_names_idx_dict(canmodel)
    entities_idx_dict = get_entities_idx_dict(canmodel)

    # Dictionaries accessing the canonical interpretation of concepts and roles
    # Keys and values are strings
    
    if INCLUDE_TOP == True:
        concept_canonical_interpretation_dict = CanonicalModel.concept_canonical_interpretation.copy()
        concept_canonical_interpretation_dict.pop('Thing')
        role_canonical_interpretation_dict = CanonicalModel.role_canonical_interpretation
    else:
        concept_canonical_interpretation_dict = CanonicalModel.concept_canonical_interpretation
        role_canonical_interpretation_dict = CanonicalModel.role_canonical_interpretation


    # Dictionary storing the domain of the canonical model being embedded
    # IMPORTANT: Keys are strings and values are CanonicalModelElements type objects
    
    domain_dict = get_domain_dict(canmodel)

    # Dictionary for easy access to entity embeddings
    # It is initialized with empty values, iteratively built by the .get_embedding_vector() method
    # Key (str): Domain Entity / Value (np.array): EntityEmbedding.embedding_vector

    entity_entityvector_dict = dict.fromkeys(domain_dict.keys())

    def __init__(self, entity_name, emb_dim, restrict_language_flag, include_top_flag, scale_factor):
        self.name = entity_name
        self.emb_dim = emb_dim
        self.scale_factor = scale_factor
        self.in_interpretation_of = []
        self.restrict_language_flag = restrict_language_flag
        self.include_top_flag = include_top_flag
        self.embedding_vector = self.get_embedding_vector()

    def get_embedding_vector(self):
        
        embedding_vector = np.zeros((self.emb_dim,))
        EntityEmbedding.entity_entityvector_dict[self.name] = []

        # Applies the embedding function to the concept names portion of the definition
        for concept_name in EntityEmbedding.concept_canonical_interpretation_dict:
            concept_name_idx = EntityEmbedding.concept_names_idx_dict[concept_name]

            if self.name in EntityEmbedding.concept_canonical_interpretation_dict[concept_name]:
                embedding_vector[concept_name_idx] = 1 * self.scale_factor
                self.in_interpretation_of.append(concept_name)

        # Applies the embedding function to the role names portion of the definition

        for role_name in EntityEmbedding.role_canonical_interpretation_dict:

            if self.restrict_language_flag == False:
                role_name_idx = len(EntityEmbedding.concept_names_idx_dict) + (EntityEmbedding.role_names_idx_dict[role_name] * len(EntityEmbedding.entities_idx_dict)) # Entities dict indexes on the domain of the canonical model

            else:
                role_name_idx = len(EntityEmbedding.concept_names_idx_dict) + EntityEmbedding.role_names_idx_dict[role_name]

            role_name_caninterp = EntityEmbedding.role_canonical_interpretation_dict[role_name]

            for pair in role_name_caninterp:

                entity_2 = pair[1] # This is a string

                if (self.name, entity_2) == pair:
                    entity_2_idx = EntityEmbedding.entities_idx_dict[entity_2]
                    final_role_entity_pair_idx = role_name_idx + entity_2_idx
                    embedding_vector[final_role_entity_pair_idx] = 1 * self.scale_factor

        EntityEmbedding.entity_entityvector_dict[self.name] = embedding_vector

        return embedding_vector
    
'''
Function for creating the binary vectors representing
each element of the domain of the canonical interpretation.

    Args:
        emb_dim (int/float): the number of dimensions of the embedding space.

    Returns:
        embedded_entities (list): a list containing all embeddings of the entities
                                  in the domain. 
    
    The embedded_entities are also available in the dictionary EntityEmbeddings.entity_entityvector_dict
'''

def get_domain_embeddings(emb_dim, restrict_language_flag, scale_factor, include_top_flag):

    embedded_entities = []
    counter = 0

   # The entities in the domain are strings
    
    for entity_name in EntityEmbedding.domain_dict:
        embedded_entity = EntityEmbedding(entity_name, emb_dim, restrict_language_flag, include_top_flag, scale_factor)
        embedded_entities.append(embedded_entity)
        counter += 1
       
        if counter % 1000 == 0:
           print(counter)
       
    return embedded_entities

'''
Final class for creating the dataset.

Inputs: concept or role names, generated
embeddings for entities in the domain of
the canonical model.

Outputs: geometrical interpretation of
concepts and role names, represented
by vertices defining a region.

One can access the GeometricInterpretation
objects either as elements in a list, or as
values in a class variable dictionary.
'''

class GeometricInterpretation:

    concept_geointerps_dict = dict.fromkeys(CanonicalModel.concept_canonical_interpretation.keys())

    if INCLUDE_TOP == True:
        concept_geointerps_dict.update({'Thing': None})

    role_geointerps_dict = dict.fromkeys(CanonicalModel.role_canonical_interpretation.keys())

    def __init__(self, name, emb_dim):
        self.name = name
        self.emb_dim = emb_dim
        self.vertices = []
        self.centroid = None

        if GeometricInterpretation.concept_geointerps_dict.get(name) is not None:
            GeometricInterpretation.concept_geointerps_dict[name] = []

    def get_centroid_naive(self):

        if self.name == 'Thing':
            centroid = np.zeros((self.emb_dim),)
            return centroid
        
        elif len(self.vertices) == 0 and self.name in self.concept_geointerps_dict.keys():
            centroid = np.zeros((self.emb_dim,))
            return centroid
        
        elif len(self.vertices) == 0 and self.name in self.role_geointerps_dict.keys():
            centroid = np.zeros((self.emb_dim * 2,)) # The centroid for the regions needs to be doubled due to the concat operation
            return centroid
        
        elif len(self.vertices) > 0 and self.name in self.concept_geointerps_dict.keys():
            n = len(self.vertices)
            centroid = np.zeros((self.emb_dim,))
            matrix = np.vstack(self.vertices)
            centroid = 1/n * np.sum(matrix, axis=0)
            return centroid
        
        elif len(self.vertices) > 0 and self.name in self.role_geointerps_dict.keys():
            n = len(self.vertices)
            centroid = np.zeros((self.emb_dim,))
            matrix = np.vstack(self.vertices)
            centroid = 1/n * np.sum(matrix, axis=0)
            return centroid
        
def index_finder(emb_dim, restrict_language_flag, concept_name_idx_dict, role_name_idx_dict, domain_idx_dict):

    index_dict = {k: None for k in range(emb_dim)}

    for k,v in concept_name_idx_dict.items():
        index_dict[v] = k

    if restrict_language_flag == False:

        for role in role_name_idx_dict:
            role_init_idx = len(concept_name_idx_dict) + (role_name_idx_dict[role] * len(domain_idx_dict))

            for entity in domain_idx_dict:
                entity_init_idx = domain_idx_dict[entity]
                final_role_entity_pair_idx = role_init_idx + entity_init_idx
                index_dict[final_role_entity_pair_idx] = (role, entity)

    elif restrict_language_flag == True:

        for role in role_name_idx_dict:
            role_init_idx = len(concept_name_idx_dict) + (role_name_idx_dict[role] * len(domain_idx_dict))

            for entity in domain_idx_dict:
                entity_init_idx = domain_idx_dict[entity]
                final_role_entity_pair_idx = role_init_idx + entity_init_idx
                index_dict[final_role_entity_pair_idx] = (role, entity)
       
    return index_dict


def get_faithful_concept_geometric_interps(concept_names_interps, domain_embeddings_list, entity_dims_index_dict, emb_dim, canmodel: CanonicalModel):

    faithful_concept_geometric_interps = []

    for concept_name in concept_names_interps.keys():
        concept_name = GeometricInterpretation(concept_name, emb_dim)

        for embedding in domain_embeddings_list:
            if concept_name.name in embedding.in_interpretation_of:
                concept_name.vertices.append(embedding.embedding_vector)
            
        GeometricInterpretation.concept_geointerps_dict[concept_name.name] = concept_name

        if concept_name.name == 'Thing':
            concept_name.centroid = np.zeros((concept_name.emb_dim,))
        else:
            concept_name.centroid = concept_name.get_centroid_naive()
        
        faithful_concept_geometric_interps.append(concept_name)

    return faithful_concept_geometric_interps


def get_faithful_role_geometric_interps(role_names_interps, entity_embeddings_list, entity_dims_index_dict, emb_dim, scaling_factor, canmodel: CanonicalModel):
    
    faithful_role_geometric_interps = []
    idx_entity_dict = entity_dims_index_dict
    relevant_idxs = len(canmodel.concept_names_dict)

    for role_name in role_names_interps.keys():
        role_name_str = role_name
        role_name = GeometricInterpretation(role_name_str, emb_dim)

        for pair in CanonicalModel.role_canonical_interpretation[role_name_str]:
            first_entity = pair[0]
            second_entity = pair[1]
            role_vertex = np.concatenate((EntityEmbedding.entity_entityvector_dict[first_entity], EntityEmbedding.entity_entityvector_dict[second_entity]))

            role_name.vertices.append(role_vertex)

        GeometricInterpretation.role_geointerps_dict[role_name_str] = role_name
        role_name.centroid = role_name.get_centroid_naive()
        faithful_role_geometric_interps.append(role_name)

    return faithful_role_geometric_interps

# The default for the scale_factor is 1 to create binary vectors.

def create_tbox_embeddings(canonical_model: CanonicalModel, restrict_language_flag: bool, include_top_flag: bool, scale_factor = 1):

    domain = canonical_model.domain # Keys are strings and values are CanonicalModelElements type objects
    concept_names_interps = canonical_model.concept_canonical_interpretation # Keys are strings and values are lists of strings.
    role_names_interps = canonical_model.role_canonical_interpretation # Keys are strings and values are lists of tuples. Tuples are of form ('C', 'D'), with C and D strings.

    SCALE_FACTOR = scale_factor
    RESTRICT_LANGUAGE = restrict_language_flag
    INCLUDE_TOP = include_top_flag
    
    if restrict_language_flag == False:

        if include_top_flag == False:
            EMB_DIM = len(concept_names_interps) + len(role_names_interps) * len(domain)
        else:
            EMB_DIM = (len(concept_names_interps) + len(role_names_interps) * len(domain))-1

        print('================EMBEDDING DIMENSION================')
        print(f'Concept Name dimensions: {len(concept_names_interps)}')
        print(f'The number of role names is: {len(role_names_interps)}')
        print(f'The size of the domain is: {len(domain)}')
        print(f'Role names dimensions: {len(role_names_interps) * len(domain)}')
        print('===================================================')
        print('')
        print(f'Final embedding dimension: {EMB_DIM}')
        print(f'The final dimension for role regions is: {EMB_DIM * 2}')

    else:

        if include_top_flag == False:
            EMB_DIM = len(concept_names_interps) + len(role_names_interps) * len(domain)
        else:
            EMB_DIM = (len(concept_names_interps) + len(role_names_interps) * len(domain))-1
        
        print('================EMBEDDING DIMENSION================')
        if include_top_flag == False:
            print(f'Concept Name dimensions: {len(concept_names_interps)}')
        else:
            print(f'Concept Name dimensions: {len(concept_names_interps)-1}')
        print(f'The number of role names is: {len(role_names_interps)}')
        print(f'The size of the domain is: {len(domain)}')
        print(f'Role names dimensions: {len(role_names_interps)}')
        print('===================================================')
        print('')
        print(f'Final embedding dimension: {EMB_DIM}')
        print(f'The final dimension for role regions is: {EMB_DIM * 2}')

    domain_embeddings_list = get_domain_embeddings(EMB_DIM, RESTRICT_LANGUAGE, SCALE_FACTOR, INCLUDE_TOP) # This function initializes the vectors with 0 and one-hot encodes them according to \mu

    concept_names_ordering = EntityEmbedding.concept_names_idx_dict
    role_names_ordering = EntityEmbedding.role_names_idx_dict
    entities_ordering = EntityEmbedding.entities_idx_dict
    
    print('')
    print('===============FINISHED EMBEDDINGS===============')
    print(f'There are {len(domain_embeddings_list)} vector embeddings.')
    print('')

    index_finder_dict = index_finder(EMB_DIM, RESTRICT_LANGUAGE, concept_names_ordering, role_names_ordering, entities_ordering)

    faithful_concept_geometric_interps = get_faithful_concept_geometric_interps(concept_names_interps, domain_embeddings_list, index_finder_dict, EMB_DIM, canonical_model)

    print('============FINISHED INTERPS CONCEPT=============')
    print(f'There are {len(faithful_concept_geometric_interps)} regions for concept names.')
    print('')

    faithful_role_geometric_interps = get_faithful_role_geometric_interps(role_names_interps, domain_embeddings_list, index_finder_dict, EMB_DIM, SCALE_FACTOR, canonical_model)

    print('=============FINISHED INTERPS ROLES==============')
    print(f'There are {len(faithful_role_geometric_interps)} regions for role names.')
    print('')

    return domain_embeddings_list, faithful_concept_geometric_interps, faithful_role_geometric_interps, index_finder_dict, int(EMB_DIM), int(scale_factor) # Returns the faithful geometric interpretations for concepts and roles as lists

domain_embeddings, concept_geointerps, role_geointerps, idx_finder_dict, EMB_DIM, SCALE_FACTOR = create_tbox_embeddings(canmodel, RESTRICT_LANGUAGE, INCLUDE_TOP, SCALE_FACTOR)