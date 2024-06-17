import torch
from torch import nn, optim
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
Computes hits@k for concept assertions.
'''

def get_hits_at_k_concept_assertions(model, GeoInterp_dataclass,
                  test_concept_assertions,
                  entity_to_idx_vocab: dict, idx_to_entity_vocab: dict,
                  idx_to_concept_vocab: dict, idx_to_role_vocab: dict,
                  centroid_score: False,
                  ):
    
    top1 = 0
    top3 = 0
    top10 = 0
    top100 = 0
    top_all = 0

    model.eval()

    hits = []

    relevant_concept_idx = []

    # Gathers only concepts appearing in the test set (it is not guaranteed that if a concept appears in the dataset, then it appears here)

    for assertion in test_concept_assertions:
        inputs, _ = assertion
        if inputs[0] not in relevant_concept_idx:
            relevant_concept_idx.append(inputs[0])
        else:
            pass

    with torch.no_grad():

        for concept_idx in relevant_concept_idx:

            assertion_scores = []

            for _, entity_idx in entity_to_idx_vocab.items():
                eval_sample = torch.tensor([concept_idx, entity_idx]).unsqueeze(0)
                outputs1, outputs2, outputs3, outputs4 = model(eval_sample) # out1 = Concept parameter, out2 = Individual parameter

                if centroid_score == False:
                    assertion_score = torch.dist(outputs1, outputs2, p=2) # Distance from the individual embedding from the concept parameter embedding
                else:
                    assertion_score = torch.dist(outputs1, outputs2, p=2) + torch.dist(outputs2, torch.tensor(GeoInterp_dataclass.concept_geointerps_dict[idx_to_concept_vocab[int(concept_idx)]].centroid)) 
                    # Distance from the individual embedding from concept param embedding plus distance from the ind emb to the centroid of the geointerp of the concept

                assertion_scores.append((torch.tensor([concept_idx, entity_idx]), assertion_score.item()))
            
            sorted_scores = sorted(assertion_scores, key=lambda x: x[1])

            k_list = [1, 3, 10, 100, len(assertion_scores)]
            hit_k_values = []

            true_samples = [inputs for inputs, _ in test_concept_assertions if inputs[0] == concept_idx] # This is problematic when dealing with big datasets

            for k in k_list:
                hit_k = any(torch.equal(scored_sample[0], true_sample) for true_sample in true_samples for scored_sample in sorted_scores[:k])
                hit_k_values.append(hit_k)

            
            hits.append(hit_k_values)

            top1 += int(hit_k_values[0])
            top3 += int(hit_k_values[1])
            top10 += int(hit_k_values[2])
            top100 += int(hit_k_values[3])
            top_all += int(hit_k_values[4])

    hits_at_k = [round(sum(hit_values) / len(hit_values), 3) for hit_values in zip(*hits)]  # Calculate hits_at_k for each k

    return hits_at_k


'''
Computes hits@k for role assertions.
'''

def get_hits_at_k_role_assertions(model, GeoInterp_dataclass,
                  test_role_assertions,
                  entity_to_idx_vocab: dict, idx_to_entity_vocab: dict,
                  idx_to_concept_vocab: dict, idx_to_role_vocab: dict,
                  centroid_score = False
                  ):
    
    top1 = 0
    top3 = 0
    top10 = 0
    top100 = 0
    top_all = 0

    model.eval()

    hits = []
    relevant_assertions = []

    # Convert PyTorch dataset to a numpy array for vectorization
    assertions_array = [assertion[0].numpy() for assertion in test_role_assertions]
    assertions_array = np.stack(assertions_array)

    '''
    The array below is used to disregard duplicate queries.
    For ex., if we have two assertions r(a,b) and r(a,c), the function
    will treat r(a, ?) as a query with b and c as positive answers. It
    will then disregard any other.
    '''

    filter_array = np.ones((assertions_array.shape), dtype=int)
    filter_counter = 0

    with torch.no_grad():

        for assertion_idx, assertion in enumerate(assertions_array):

            filter_counter = assertion_idx

            if np.all(filter_array[filter_counter] == 1):

                head_entity_idx = assertion[0]
                role_entity_idx = assertion[1]
                filter_arr = (assertions_array[:, 0] == head_entity_idx) & (assertions_array[:, 1] == role_entity_idx)
                relevant_assertions_idcs = np.where(filter_arr)[0]
                relevant_assertions = torch.tensor(np.array([assertions_array[idx] for idx in relevant_assertions_idcs]))
                filter_array[relevant_assertions_idcs] = 0

                assertion_scores = []

                for _, tail_entity_idx in entity_to_idx_vocab.items():
                    eval_sample = torch.tensor([head_entity_idx, role_entity_idx, tail_entity_idx]).unsqueeze(0)
                    outputs1, outputs1_2, outputs2, outputs3, _ = model(eval_sample)
                    real_concat = torch.cat((outputs2, outputs3), dim=1)
                    role_centroid = torch.tensor(GeoInterp_dataclass.role_geointerps_dict[idx_to_role_vocab[role_entity_idx]].centroid)

                    if centroid_score == False:
                        assertion_score = torch.dist(outputs1, real_concat, p=2)                                                                           
                    else:
                        assertion_score = torch.dist(real_concat, role_centroid) + torch.dist(torch.cat((outputs2, outputs1_2), dim=1), role_centroid) + torch.dist(torch.cat((outputs1, outputs3), dim=1), role_centroid) + torch.dist(torch.cat((outputs1, outputs1_2), dim=1), role_centroid)

                    assertion_scores.append((torch.tensor([head_entity_idx, role_entity_idx, tail_entity_idx]), assertion_score.item()))



                sorted_scores = sorted(assertion_scores, key=lambda x: x[1])

                k_list = [1, 3, 10, 100, len(assertion_scores)]
                hit_k_values = []

                for k in k_list:
                    hit_k = any(torch.equal(scored_sample[0], assertion) for assertion in relevant_assertions for scored_sample in sorted_scores[:k])
                    hit_k_values.append(hit_k)
            
                hits.append(hit_k_values)

                top1 += int(hit_k_values[0])
                top3 += int(hit_k_values[1])
                top10 += int(hit_k_values[2])
                top100 += int(hit_k_values[3])
                top_all += int(hit_k_values[4])

            else:
                pass
    
    hits_at_k = [round(sum(hit_values) / len(hit_values), 3) for hit_values in zip(*hits)]  # Calculate hits_at_k for each k

    return hits_at_k

'''
Helper function to plot the loss after training a model.
'''

def plot_loss(train_loss, test_loss, num_epoch):
    
    plt.plot(range(1, num_epoch+1), train_loss, 'b-', label='Train Loss')
    plt.plot(range(1, num_epoch+1), test_loss, 'r-', label='Test Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss per Epoch')
    plt.legend()

    # Display the plot
    plt.show()


'''
Helper function to save models hparams and scores as a dictionary.
'''
def save_model(centroid_score, lr,
               phi, gamma, psi, emb_dim, epochs,
               log_epoch, eval_freq,
               eval_test,
               loss_fn, model, optimizer,
               train_loss_list, test_loss_list,
               train_hits_at_k_concept, test_hits_at_k_concept,
               train_hits_at_k_role, test_hits_at_k_role):
    
    model_hparams = {'centroid_score': centroid_score,
                     'lr': lr,
                     'phi': phi,
                     'gamma': gamma,
                     'psi': psi,
                     'emb_dim': emb_dim,
                     'epochs': epochs,
                     'log_epoch': log_epoch,
                     'eval_freq': eval_freq,
                     'eval_test': eval_test,
                     'loss_fn': loss_fn,
                     'model': model,
                     'optimizer': optimizer,
                     'train_loss_list': train_loss_list,
                     'test_loss_list': test_loss_list,
                     'train_hits_at_k_concept': train_hits_at_k_concept,
                     'test_hits_at_k_concept': test_hits_at_k_concept,
                     'train_hits_at_k_role': train_hits_at_k_role,
                     'test_hits_at_k_role': test_hits_at_k_role,
                     'misc notes': []
                     }
    
    return model_hparams

def plot_score_hak(hits_at_k_concept, hits_at_k_roles, topk, num_epoch, eval_freq):

    concept_hits_at_topk = [score_list[topk] for score_list in hits_at_k_concept]
    roles_hits_at_topk = [scores[topk] for scores in hits_at_k_roles]

    hak_dict = {0: 1,
                1: 3,
                2: 10,
                3: 100,
                4: 'all'}
    
    plt.plot(range(1, num_epoch+1, eval_freq), concept_hits_at_topk, 'b-', label=f'H@{hak_dict[topk]} concepts')

    try:
        plt.plot(range(1, num_epoch+1, eval_freq), roles_hits_at_topk, 'r-', label=f'H@{hak_dict[topk]} roles')
    except:
        print('No roles to plot.')

    plt.ylim(0, 1.02)
    plt.xlabel('Epochs')
    plt.ylabel(f'hits@{hak_dict[topk]}')
    plt.title(f'Hits@{hak_dict[topk]} every {eval_freq} epochs')
    plt.legend()

    plt.show()

'''
Helper function for representing two dimensions of the models params graphically.
'''

def plot_model(model, GeoInterp_dataclass, individual_vocab_idcs, concept_vocab_idcs, role_vocab_idcs, scaling_factor, dim1, dim2):

    individual_embeddings = model.individual_embedding_dict.weight
    concept_parameter_embeddings = model.concept_embedding_dict.weight
    role_parameter_embeddings = torch.cat((torch.tensor(model.role_head_embedding_dict.weight), torch.tensor(model.role_tail_embedding_dict)), dim=1)

    individuals_for_plotting = []
    concept_parameters_for_plotting = []
    concept_centroid_for_plotting = []
    role_parameters_for_plotting = []
    role_centroid_for_plotting = []

    for idx, individual in enumerate(individual_embeddings[:]):
        individual = individual[:].detach().numpy()
        individual_label = individual_vocab_idcs[idx]
        final_representation = (individual, individual_label)
        individuals_for_plotting.append(final_representation)

    for idx, concept in enumerate(concept_parameter_embeddings):
        concept_param = concept[:].detach().numpy()
        concept_label = concept_vocab_idcs[idx]
        final_representation = (concept_param, concept_label)
        concept_parameters_for_plotting.append(final_representation)

    for idx, key in enumerate(GeoInterp_dataclass.concept_geointerps_dict.keys()):
        concept_centroid = GeoInterp_dataclass.concept_geointerps_dict[key].centroid[:]
        concept_label = key + '_centroid'
        final_representation = (concept_centroid, concept_label)
        concept_centroid_for_plotting.append(final_representation)

    for idx, role in enumerate(role_parameter_embeddings):
        role_param = role[:].detach().numpy()
        role_label = role_vocab_idcs[idx]
        final_representation = (role_param, role_label)
        role_parameters_for_plotting.append(final_representation)

    for idx, key in enumerate(GeoInterp_dataclass.role_geointerps_dict.keys()):
        role_centroid = GeoInterp_dataclass.role_geointerps_dict[key].centroid[:]
        role_label = key + '_centroid'
        final_representation = (role_centroid, role_label)
        role_centroid_for_plotting.append(final_representation)


    # Create a figure and axis object
    fig, ax = plt.subplots()

    ax.set_xlim(-1, scaling_factor + scaling_factor/10)
    ax.set_ylim(-1, scaling_factor + scaling_factor/10)
    ax.grid(True)

    ax.plot(0, 0, 'yo')

    # Plot individual points in blue
    for individual, label in individuals_for_plotting:
        ax.plot(individual[dim1], individual[dim2], 'bo', label=label)
        ax.annotate(label, xy=(individual[dim1], individual[dim2]), xytext=(3, -3), textcoords='offset points')

    # Plot concept points in red
    for concept_param, label in concept_parameters_for_plotting:
        ax.plot(concept_param[dim1], concept_param[dim2], 'r+', label=label)
        ax.annotate(label, xy=(concept_param[dim1], concept_param[dim2]), xytext=(3, -3), textcoords='offset points')

    for concept_centroid, label in concept_centroid_for_plotting:
        ax.plot(concept_centroid[dim1], concept_centroid[dim2], 'go', label=label)
        ax.annotate(label, xy=(concept_centroid[dim1], concept_centroid[dim2]), xytext=(3, -3), textcoords='offset points')

    # Plot role points in yellow
    
    for role_param, label in role_parameters_for_plotting:
        ax.plot(role_param[dim1], role_param[dim2], 'y+', label=label)
        ax.annotate(label, xy=(role_param[dim1], role_param[dim2]), xytext=(3, -3), textcoords='offset points')
        
    for role_centroid, label in role_centroid_for_plotting:
        ax.plot(role_centroid[dim1], role_centroid[dim2], 'yo', label=label)
        ax.annotate(label, xy=(role_centroid[dim1], role_centroid[dim2]), xytext=(3, -3), textcoords='offset points')

    plt.show()


''' 
Helper function for plotting H@K evals per increment.
'''

def plot_score_hak(hits_at_k_concept, hits_at_k_roles, topk, num_epoch, eval_freq):

    concept_hits_at_topk = [score_list[topk] for score_list in hits_at_k_concept]
    roles_hits_at_topk = [scores[topk] for scores in hits_at_k_roles]

    hak_dict = {0: 1,
                1: 3,
                2: 10,
                3: 100,
                4: 'all'}
    
    plt.plot(range(1, num_epoch+1, eval_freq), concept_hits_at_topk, 'b-', label=f'H@{hak_dict[topk]} concepts')

    try:
        plt.plot(range(1, num_epoch+1, eval_freq), roles_hits_at_topk, 'r-', label=f'H@{hak_dict[topk]} roles')
    except:
        print('No roles to plot.')

    plt.ylim(0, 1.02)
    plt.xlabel('Epochs')
    plt.ylabel(f'hits@{hak_dict[topk]}')
    plt.title(f'Hits@{hak_dict[topk]} every {eval_freq} epochs')
    plt.legend()

    plt.show()

'''
Helper function for computing concept assertion loss.
    Args:
        outputs1: The concept parameter;
        outputs2: The individual's parameter;
        outputs3: The negatively sampled individual:
        labels: the centroid of the concept's geometric interpretation
        loss_fn: selected metric, standard is elementwise-MSE
        gamma = Hyperparameter for weighting how far the individual parameters are allowed to move;
        phi: Hyperparameter for weighting how far the role parameter is allowed to move;
        neg_sampling: Flag for indicating whether negative sampling is used or not.
'''

def compute_loss_concept(outputs1, outputs2, outputs3, labels, loss_fn, gamma, phi, psi, neg_sampling):
    
    if neg_sampling:
        d_indparam_centroid = loss_fn(outputs2, labels)
        d_conceptparam_indparam = gamma * loss_fn(outputs1, outputs2)
        d_conceptparam_centroid = phi * loss_fn(outputs1, labels)
        d_indparam_negsamparam = psi * -loss_fn(outputs2.detach(), outputs3)

        loss = d_indparam_centroid + d_conceptparam_indparam + d_conceptparam_centroid + d_indparam_negsamparam

    else:
        loss = loss_fn(outputs2, labels) + gamma * loss_fn(outputs1, outputs2) + phi * loss_fn(outputs1, labels)

    return loss

'''
Helper function for computing role assertion loss.
Args:
    outputs1 = Role parameter;
    outputs2 = Subject entity parameter;
    outputs3 = Head entity parameter;
    outputs4 = Neg sampled entity parameter;
    labels = Centroid for the role's geometric interpretation;
    loss_fn = The desired metric, testing with MSELoss;
    gamma = Hyperparameter for weighting how far the individual parameters are allowed to move;
    phi: Hyperparameter for weighting how far the role parameter is allowed to move;
    neg_sampling: Flag for indicating whether negative sampling is used or not.

The loss term is defined as the sum of the a) distance between the concat between (H,T) and the centroid of the role
                                                    b) gamma-weighted distance between role parameter and (H,T) concat
                                                    c) phi-weighted distance between role parameter and centroid
                                                    d) negative distance between (H,T) concat and (H, Corrupt T) concat
                                                    e) negative distance between (H,T) concat and (Corrupt H, T) concat
'''

def compute_loss_role(outputs1, outputs1_2, outputs2, outputs3, outputs4, labels, loss_fn, gamma, phi, psi, neg_sampling):
    
    real_concat_no_detach = torch.cat((outputs2, outputs3), dim=1)
    real_concat_both_detach = torch.cat((outputs2.detach(), outputs3.detach()), dim=1)
    real_concat_tail_detach = torch.cat((outputs2, outputs3.detach()), dim=1)
    real_concat_head_detach = torch.cat((outputs2.detach(), outputs3), dim=1)
    tail_corrupted_concat = torch.cat((outputs2.detach(), outputs4), dim=1)
    head_corrupted_concat = torch.cat((outputs4, outputs3.detach()), dim=1)

    if neg_sampling:
        
        loss = (loss_fn(real_concat_no_detach, labels) +
                gamma * loss_fn(torch.cat((outputs1, outputs3), dim=1), real_concat_no_detach) +
                gamma * loss_fn(torch.cat((outputs2, outputs1_2), dim=1), real_concat_no_detach) +
                phi * loss_fn(torch.cat((outputs1, outputs1_2), dim=1), labels) +
                - (psi * loss_fn(real_concat_both_detach, tail_corrupted_concat))
                )
        
    else:
        loss = (loss_fn(real_concat_no_detach, labels) +
                gamma * loss_fn(outputs1, real_concat_no_detach) +
                phi * loss_fn(outputs1, labels))

    return loss

'''
Main training loop.
'''

def train_concept(model, concept_dataloader, loss_fn, optimizer, neg_sampling: bool):
    model.train()
    concept_loss = 0.0
    total_samples = 0

    for i, data in enumerate(concept_dataloader):

        inputs, labels = data

        optimizer.zero_grad()
        outputs1, outputs2, outputs3, _ = model(inputs) # Outputs 1 = Concept Parameter, Outputs 2 = Entity Parameter, Outputs 3 = neg_candidate, out4 = None
        loss = compute_loss_concept(outputs1, outputs2, outputs3, labels, loss_fn, model.gamma, model.phi, model.psi, neg_sampling)

        loss.backward()
        optimizer.step()
        concept_loss += loss.item()
        samples_num = inputs.size(0)
        total_samples += samples_num

        model.concept_parameter_constraint()

    return concept_loss, total_samples

def train_role(model, role_dataloader, loss_fn, optimizer, neg_sampling: bool):

    model.train()
    role_loss = 0.0
    total_samples = 0

    for i, data in enumerate(role_dataloader):
        model.train()
        inputs, labels = data
        optimizer.zero_grad()
        outputs1, outputs1_2, outputs2, outputs3, outputs4 = model(inputs) # Outputs1 = Role Parameter, outputs2 = SubjEntity, out3 = HeadEntity, outputs4 = neg_candidate

        loss = compute_loss_role(outputs1, outputs1_2, outputs2, outputs3, outputs4, labels, loss_fn, model.gamma, model.phi, model.psi, neg_sampling)
        
        loss.backward()
        optimizer.step()
        role_loss += loss.item()
        samples_num = inputs.size(0)
        total_samples += samples_num

        model.role_parameter_constraint()

    return role_loss, total_samples

def train(model, concept_dataloader, role_dataloader, loss_fn, optimizer, neg_sampling: bool, train_category: bool):
    model.train()

    if train_category == 0:
        role_loss, role_samples = train_role(model, role_dataloader, loss_fn, optimizer, neg_sampling)
        concept_loss, concept_samples = train_concept(model, concept_dataloader, loss_fn, optimizer, neg_sampling)
        total_loss = concept_loss + role_loss
        num_samples = concept_samples + role_samples
        return total_loss / num_samples
    
    elif train_category == 1:
        concept_loss, concept_samples = train_concept(model, concept_dataloader, loss_fn, optimizer, neg_sampling)
        return concept_loss / concept_samples

    elif train_category == 2:
        role_loss, role_samples = train_role(model, role_dataloader, loss_fn, optimizer, neg_sampling)
        return role_loss / role_samples

''' 
Functions for obtaining test loss.
'''

def test_concept_loss(model, concept_dataloader, loss_fn, neg_sampling: bool):
    
    model.eval()
    test_concept_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for i, data in enumerate(concept_dataloader):
            inputs, labels = data
            outputs1, outputs2, outputs3, outputs4 = model(inputs) # Outputs1 = Role Parameter, outputs2 = SubjEntity, out3 = HeadEntity, outputs4 = neg_candidate
            
            loss = compute_loss_concept(outputs1, outputs2, outputs3, labels, loss_fn, model.gamma, model.phi, model.psi, neg_sampling)

            test_concept_loss += loss.item()
            samples_num = inputs.size(0)
            total_samples += samples_num

    return test_concept_loss, total_samples

def test_role_loss(model, role_dataloader, loss_fn, neg_sampling: bool):

    model.eval()
    test_role_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for i, data in enumerate(role_dataloader):
            inputs, labels = data
            outputs1, outputs1_2, outputs2, outputs3, outputs4 = model(inputs) # Outputs1 = Role Parameter, outputs2 = SubjEntity, out3 = HeadEntity, outputs4 = neg_candidate
            
            loss = compute_loss_role(outputs1, outputs1_2, outputs2, outputs3, outputs4, labels, loss_fn, model.gamma, model.phi, model.psi, neg_sampling)

            test_role_loss += loss.item()
            samples_num = inputs.size(0)
            total_samples += samples_num

    return test_role_loss, total_samples

'''
Function call for obtaining test loss for concept and role assertions sequentially.
'''
def test(model, concept_dataloader, role_dataloader, loss_fn, neg_sampling: bool, train_category: int):
    model.eval()
    total_loss = 0.0

    if train_category == 0:
        
        role_test_loss, role_samples = test_role_loss(model, role_dataloader, loss_fn, neg_sampling)
        concept_test_loss, concept_samples = test_concept_loss(model, concept_dataloader, loss_fn, neg_sampling)
        total_samples = concept_samples + role_samples
        total_loss = concept_test_loss + role_test_loss
        return total_loss / total_samples
    
    elif train_category == 1:
        concept_test_loss, concept_samples = test_concept_loss(model, concept_dataloader, loss_fn, neg_sampling)
        return concept_test_loss / concept_samples
    
    elif train_category == 2:
        role_test_loss, role_samples = test_role_loss(model, role_dataloader, loss_fn, neg_sampling)
        return role_test_loss / role_samples

'''
Main function for training.
GeoInterp_dataclass: we need to pass the GeometricInterpretation class to the model, to the evaluation functions, and to the plotting functions.
'''

def train_model(model, GeoInterp_dataclass,
                train_concept_loader, train_role_loader,
                test_concept_loader, test_role_loader,
                train_concept_dset, test_concept_dset,
                train_role_dset, test_role_dset,
                num_epochs, loss_log_freq,
                eval_freq, eval_train,
                loss_function, optimizer,
                idx_to_entity: dict, entity_to_idx: dict,
                idx_to_concept: dict, concept_to_idx: dict,
                idx_to_role: dict, role_to_idx: dict,
                centroid_score: bool, neg_sampling: bool,
                plot_loss_flag: bool, train_category: int
                ):

    train_loss_list = []
    test_loss_list = []

    train_hits_at_k_concept = []
    test_hits_at_k_concept = []

    train_hits_at_k_role = []
    test_hits_at_k_role = []

    
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_concept_loader, train_role_loader, loss_function, optimizer, neg_sampling, train_category)
        train_loss_list.append(train_loss)
        test_loss = test(model, test_concept_loader, test_role_loader, loss_function, neg_sampling, train_category)
        test_loss_list.append(test_loss)

        if epoch % loss_log_freq == 0:
            print(f'Epoch {epoch}/{num_epochs} -> Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}\n')

        if epoch % eval_freq == 0 or epoch == 0:
            print(f'Epoch {epoch}: Initiating evaluation. \n')

            try:
                test_hak_concept = get_hits_at_k_concept_assertions(model, GeoInterp_dataclass, test_concept_dset, entity_to_idx, idx_to_entity, idx_to_concept, idx_to_role, centroid_score)
                test_hits_at_k_concept.append(test_hak_concept)
            except:
                print('Exception found. H@K for the Concept Test Dataset have not been computed.')
                pass

            if eval_train == True:
                try:
                    if train_category == 0 or 2:
                        train_hak_concept = get_hits_at_k_concept_assertions(model, GeoInterp_dataclass, train_concept_dset, entity_to_idx, idx_to_entity, idx_to_concept, idx_to_role, centroid_score)
                        train_hits_at_k_concept.append(train_hak_concept)
                except:
                    print('Exception found. H@K for the Concept Train Dataset have not been computed.')
                    pass
            
            try:
                test_hak_role = get_hits_at_k_role_assertions(model, GeoInterp_dataclass, test_role_dset, entity_to_idx, idx_to_entity, idx_to_concept, idx_to_role, centroid_score)
                test_hits_at_k_role.append(test_hak_role)
            except:
                print('Exception found. H@K for the Role Test Dataset have not been computed.')
                pass
            
            if eval_train == True:
                try:
                    train_hak_role = get_hits_at_k_role_assertions(model, GeoInterp_dataclass, train_role_dset, entity_to_idx, idx_to_entity, idx_to_concept, idx_to_role, centroid_score)
                    train_hits_at_k_role.append(train_hak_role)
                except:
                    print('Exception found. H@K for the Role Train Dataset have not been computed.')
                    pass
    
    if plot_loss_flag == True:
        plot_loss(train_loss_list, test_loss_list, num_epochs)

    return train_loss_list, test_loss_list, train_hits_at_k_concept, test_hits_at_k_concept, train_hits_at_k_role, test_hits_at_k_role