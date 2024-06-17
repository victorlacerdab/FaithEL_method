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

# Change the directory.
dir = '/Users/victorlacerda/Desktop/family_ontology.owl' 

RESTRICT_LANGUAGE = False # This option is deprecated. Do not change to True.
INCLUDE_TOP = True

'''
Class for creating entities to
populate the creation of the
canonical models.
The .name attribute is used to
create a single representation
for concepts like A and B / 
B and A, as they are the same.
'''

class CanonicalModelElements:

    concept_names = {}
    concept_intersections = {}
    concept_restrictions = {}
    all_concepts = {}

    def __init__(self, concept):
        self.concept = concept
        self.name = self.get_name()
        self.get_element_dict()

    def get_name(self):
        
        if type(self.concept) == ThingClass:
            return self.concept.name

        elif type(self.concept) == Restriction:
            return 'exists_' + self.concept.property.name + '.' + self.concept.value.name
        
        else:
            return 'And_' + ''.join(sorted(self.concept.Classes[0].name + self.concept.Classes[1].name)) # The name is sorted to avoid that (e.g) (A \and B) and (B \and A) are treated as different concepts
        
    def get_element_dict(self):

        if type(self.concept) == ThingClass:
            CanonicalModelElements.concept_names[self.name] = self
            CanonicalModelElements.all_concepts[self.name] = self

        elif type(self.concept) == Restriction:
            CanonicalModelElements.concept_restrictions[self.name] = self
            CanonicalModelElements.all_concepts[self.name] = self

        elif type(self.concept) == And:
            CanonicalModelElements.concept_intersections[self.name] = self
            CanonicalModelElements.all_concepts[self.name] = self

def get_canonical_model_elements(concept_names_iter, role_names_iter, ontology, restrict_language = False, include_top = True):
    
    onto = ontology

    if include_top == True:
        top = owl.Thing
        CanonicalModelElements(top)
        #bottom = owl.Nothing
        #CanonicalModelElements(bottom)

    for concept_name in concept_names_iter:
        CanonicalModelElements(concept_name)

        if restrict_language == False:

            for concept_name2 in concept_names_iter:
        
                with onto:
                    gca = GeneralClassAxiom(concept_name & concept_name2)
                    gca.is_a.append(concept_name & concept_name2)
            
                CanonicalModelElements(gca.left_side)

    print('All Concept Names and Concept Intersections have been preprocessed for the creation of the canonical model.')
    print('===========================================================================================================')

    if include_top == True:
        concept_names_iter.append(top)
        #concept_names_iter.append(bottom)
    else:
        print('Top is not being included in the canonical model.')
        pass

    if restrict_language == False:
        for role_name in role_names_iter:
            for concept_name in concept_names_iter:
                with onto:
                    gca = GeneralClassAxiom(role_name.some(concept_name))
                    gca.is_a.append(role_name.some(concept_name))

                CanonicalModelElements(gca.left_side)
    
    else:
        if include_top == True:
            for role_name in role_names_iter:
                with onto:
                    gca = GeneralClassAxiom(role_name.some(owl.Thing))
                    gca.is_a.append(role_name.some(owl.Thing))
                    CanonicalModelElements(gca.left_side)
        else:
            print(f'Top must be included when restricting the language. Terminating the creation of the canonical model.')
            sys.exit(1)
            
    print('')
    print('All restrictions have been preprocessed for the creation of the canonical model.')


'''
The main class for creating the canonical model for the ontology.

The canonical model is stored in dictionaries available as class variables 'concept_canonical_interpretation'
and 'role_canonical_interpretation'. 

Args:
    concept_names_dict: a dictionary stored in the CanonicalModelElement class.
    concept_intersection_dict: a dictionary stored in the CanonicalModelElement class.
    concept_restrictions_dict: a dictionary stored in the CanonicalModelElement class.
    all_concepts_dict: a dictionary stored in the CanonicalModelElement class.
    role_names_iter (list): a list containing all role names in the loaded ontology.
'''

class CanonicalModel:

    concept_canonical_interpretation = {}
    role_canonical_interpretation = {}

    def __init__(self, concept_names_dict, concept_intersections_dict, concept_restrictions_dict, all_concepts_dict, role_names_iter, include_top_flag):
        
        self.domain = all_concepts_dict
        self.concept_names_dict = concept_names_dict
        self.concept_restrictions_dict = concept_restrictions_dict
        self.concept_intersections_dict = concept_intersections_dict
        self.include_top = include_top_flag

        self.role_names_iter = role_names_iter

        self.concept_canonical_interp = self.get_concept_name_caninterp() # These are only used to build the concept_canonical_interpretation and role_canonical_interpretation class attributes
        self.role_canonical_interp = self.get_role_name_caninterp()       # The functions do not return anything, they just update their corresponding class variables 


    def get_concept_name_caninterp(self):

        for concept in self.concept_names_dict.keys():
            CanonicalModel.concept_canonical_interpretation[concept] = []

        for concept in self.domain.keys():

            if self.include_top == True:
                superclasses = self.domain[concept].concept.ancestors(include_self=True, include_constructs=False)
                superclasses = [superclass for superclass in superclasses if type(superclass) == ThingClass]

                for superclass in superclasses:
                    if superclass.name != 'Thing':
                        CanonicalModel.concept_canonical_interpretation[superclass.name].append(concept)
                        CanonicalModel.concept_canonical_interpretation[superclass.name].append('And_' + ''.join(sorted(concept + concept))) # Hardcoded to cope with owlready2

                CanonicalModel.concept_canonical_interpretation['Thing'] = list(set([concept_name for concept_name in self.domain.keys()]))
            
            else:
                superclasses = list(self.domain[concept].concept.ancestors(include_self=True, include_constructs=False))
                superclasses = [superclass for superclass in superclasses if type(superclass) == ThingClass and superclass.name != 'Thing']

                for superclass in superclasses:
                    if superclass.name != 'Thing':
                        CanonicalModel.concept_canonical_interpretation[superclass.name].append(concept)

        ''' 
        This section is added to hardcode elements to the canonical interpretation of 
        concept names which are not correctly captured by the Owl2Ready library.
        '''

        for concept in self.concept_names_dict.keys():
            if 'Q5' in concept:
                pass
            else:
                CanonicalModel.concept_canonical_interpretation[concept].append('And_' + ''.join(sorted(concept + 'Q5')))
            
    def get_role_name_caninterp(self):

        # Initialize the dictionary storing the canonical interpretation of roles

        for role_name in self.role_names_iter:

            role_name_str = role_name.name # Accesses the property type object's name as a string
            CanonicalModel.role_canonical_interpretation[role_name_str] = []

        # First case from Definition 10

        for role_name in self.role_names_iter:
            superroles = role_name.ancestors(include_self=True)
            superroles = [superrole for superrole in superroles if superrole.name not in ['topObjectProperty', 'ObjectProperty', 'Property', 'SymmetricProperty']]

            for superrole in superroles:
                for restriction_name, restriction_concept in self.concept_restrictions_dict.items():
                    if role_name == restriction_concept.concept.property:
                        concept_name_str = restriction_concept.concept.value.name
                        pair = (restriction_name, concept_name_str)
                        CanonicalModel.role_canonical_interpretation[superrole.name].append(pair)
                
        # Second case from Definition 10            
        for concept_name in self.concept_names_dict.keys():
            if concept_name == 'Thing':
                pass
            else:
                concept_name = self.concept_names_dict[concept_name].concept
                superclasses = concept_name.ancestors(include_self=False, include_constructs=True)
                superclasses = [supercls for supercls in superclasses if type(supercls) == Restriction]

                for superclass in superclasses:
                    if type(superclass) == Restriction:
                        superclass_concept_value = superclass.value
                        superclass_concept_role = superclass.property
                        c_B = superclass_concept_value.name
                    else:
                        pass

                    for concept in self.concept_names_dict.keys():
                        if concept == 'Thing':
                            pass
                        else:
                            concept = self.concept_names_dict[concept].concept
                            ancestors = concept.ancestors(include_self=True, include_constructs=False)
                            if concept_name in ancestors:
                                c_D = concept.name
                                self.role_canonical_interpretation[superclass_concept_role.name].append((c_D, c_B))
                                c_D = 'And_' + ''.join(sorted(c_D + c_D))
                                self.role_canonical_interpretation[superclass_concept_role.name].append((c_D, c_B))
                            
                    for concept in self.concept_restrictions_dict.keys():
                        concept = self.concept_restrictions_dict[concept].concept
                        ancestors = concept.ancestors(include_self=False, include_constructs=False)
                        if concept_name in ancestors:
                            c_D = 'exists_' + superclass.property.name + '.' + superclass.value.name
                            self.role_canonical_interpretation[superclass_concept_role.name].append((c_D, c_B))

                    for concept in self.concept_intersections_dict.keys():
                        concept = self.concept_intersections_dict[concept].concept
                        ancestors = concept.ancestors(include_self=False, include_constructs=True)
                        if concept_name in ancestors:
                            c_D = 'And_' + ''.join(sorted(superclass.Classes[0].name + superclass.Classes[1].name))
                            self.role_canonical_interpretation[superclass_concept_role.name].append((c_D, c_B))

'''
Main function for creating the canonical model.

    Args:
        onto_dir (str): a string pointing to the directory where the ontology is stored.

    Returns:
        canmodel (CanonicalModel): returns a variable containing the canonical model. 
        
        Attention: the interpretations of concept names and role names can also be accessed via class variables
        from the CanonicalModel class.
'''

def create_canonical_model(onto_dir: str, restrict_language_flag: bool, include_top_flag: bool):

    onto = get_ontology(onto_dir)
    onto = onto.load()

    individuals_iter = list(onto.individuals())
    gcas_iter = list(onto.general_class_axioms()) # Attention: this will not work unless the generator is converted into a list
    concept_names_iter = list(onto.classes())
    role_names_iter = list(onto.properties())

    get_canonical_model_elements(concept_names_iter, role_names_iter, onto, restrict_language_flag, include_top_flag)

    print('============================================================================')
    print('Starting to reason.')

    with onto:
        sync_reasoner(debug=0)

    gcas_iter = list(onto.general_class_axioms()) # Attention: this will not work unless the generator is converted into a list
    concept_names_iter = list(onto.classes())
    role_names_iter = list(onto.properties())
    individuals_iter = list(onto.individuals())

    print('')
    print('============================================================================')
    print('Done reasoning. Creating the canonical model.')
    canmodel = CanonicalModel(CanonicalModelElements.concept_names, CanonicalModelElements.concept_intersections, CanonicalModelElements.concept_restrictions, CanonicalModelElements.all_concepts, role_names_iter, INCLUDE_TOP)
    print('============================================================================\n')
    print('Concluded creating canonical model.')

    return canmodel

# Instantiates the canonical model

canmodel = create_canonical_model(dir, RESTRICT_LANGUAGE, INCLUDE_TOP)