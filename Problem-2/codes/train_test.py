from models import GANDiscriminatorModel, GANGeneratorModel
from preprocessing import eICUDataSet, MakeTrainOrTestData
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAccuracy
from tboard import Tboard, make_image_plot, Write_FullInfo
from torch import optim
import torch.nn as nn
import torch
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

def TestModel(config: dict, generator: GANGeneratorModel, discriminator: GANDiscriminatorModel, tensorboardSummary = None):
    # hardware information ###############

    device = config["device"]

    # tensorboard information ############

    test_dataset = eICUDataSet(config["test_dataset"])
    Integral = test_dataset.Integral
    Integral = 1.

    # Output information #################

    out_path = os.path.join(config["output"]["dir"], config["name"] + "_" + config["plot_title"])
    if(os.path.exists(out_path) == False):
        print ("cannot find output folder")

    testing_data = MakeTrainOrTestData(test_dataset, config["testing"])

    discriminator.to(device)
    generator.to(device)

    # Testing information ##############

    batch_size = config["testing"]["batch_size"]
    n_evals = config["testing"]["n_evals"]
    zn = config["testing"]["latent_dim"]
    true_input_dim = config["testing"]["input_data_len"]

    # Binary Cross Entropy loss #########
    loss = nn.BCELoss()

    # metrics to keep track of ##########
    bin_acc = BinaryAccuracy()
    acc = Accuracy()

    # Reset metrics for every epoch.
    D_loss = []
    D_acc_true = []
    D_acc_gen = []

    # Metric that is very interesting
    D_loss_true = []
    D_loss_gen = []

    for batch_idx, data_input in enumerate(testing_data):

        noise = torch.randn(batch_size, zn).to(device)
        generated_data = generator(noise) # batch_size X 76
        # Discriminator
        true_data = data_input[0].view(batch_size, true_input_dim).to(device) # batch_size X 76
        digit_labels = data_input[1] # batch_size
        true_labels = torch.ones(batch_size).to(device)

        discriminator_output_for_true_data = discriminator(true_data.float()).view(batch_size)
        true_discriminator_loss = loss(discriminator_output_for_true_data, true_labels)

        discriminator_output_for_generated_data = discriminator(generated_data.detach()).view(batch_size)
        generator_discriminator_loss = loss(
                                        discriminator_output_for_generated_data, torch.zeros(batch_size).to(device)
                                        )

        discriminator_acc_for_true_data = bin_acc(discriminator_output_for_true_data.cpu(), true_labels.cpu())
        D_acc_true.append(discriminator_acc_for_true_data)
        discriminator_acc_for_generated_data = bin_acc(discriminator_output_for_generated_data.cpu(), torch.zeros(batch_size).cpu())
        D_acc_gen.append(discriminator_acc_for_generated_data)


        D_loss_true.append(true_discriminator_loss.data.item())

        D_loss_gen.append(generator_discriminator_loss.data.item())

        with torch.no_grad():
            noise = torch.randn(batch_size,zn).to(device)
            generated_data = generator(noise).cpu().view(batch_size, true_input_dim)
            tensorboardSummary.writer.add_histogram("Test/difference",
                                                    (generated_data - true_data.cpu()).flatten()*Integral)
            tensorboardSummary.writer.flush()
            generated_data = generated_data[0::4].reshape(5, 5, true_input_dim)
            pdata = true_data[0::4].cpu().reshape(5, 5, true_input_dim)
            fig = make_image_plot(generated_data, pdata, Integral, true_input_dim)
            tensorboardSummary.writer.add_figure(config["plot_title"] + "/Test", fig[0], global_step = batch_idx)
            tensorboardSummary.writer.flush()

    tensorboardSummary.writer.add_scalar('Test/Loss/Discrim_loss_true', torch.mean(torch.FloatTensor(D_loss_true)), batch_idx)
    tensorboardSummary.writer.add_scalar('Test/Loss/Gen_loss_gen', torch.mean(torch.FloatTensor(D_loss_gen)), batch_idx)
    tensorboardSummary.writer.add_scalar('Test/Accuracy/Discrim_true_acc', torch.mean(torch.FloatTensor(D_acc_true)), batch_idx)
    tensorboardSummary.writer.add_scalar('Test/Accuracy/Discrim_gen_acc', torch.mean(torch.FloatTensor(D_acc_gen)), batch_idx)

    tensorboardSummary.writer.add_text('Test/', "Test is done with 20% Down Sampling")
    tensorboardSummary.writer.flush()

def TrainModel(config: dict, load_chckpt: bool = False):
    # hardware information ###############

    device = config["device"]

    # tensorboard information ############

    tensorboardSummary = Tboard(config)

    dataset = eICUDataSet(config["dataset"])
    Integral = dataset.Integral
    Integral = 1.

    # Output information #################

    out_path = os.path.join(config["output"]["dir"], config["name"] + "_" + config["plot_title"])
    if(os.path.exists(out_path) == False):
        os.makedirs(out_path)

    # training_data -> batchsize x 76 x 2
    training_data = MakeTrainOrTestData(dataset, config["training"])


    # size of input_vector -> 76
    config["training"]["input_data_len"] = dataset.dim[0]

    discriminator = GANDiscriminatorModel(config)
    generator = GANGeneratorModel(config)

    if(load_chckpt):
        if(os.path.exists(config["output"]["gen_chckpt_file"]) and os.path.exists(config["output"]["dis_chckpt_file"])):
            discriminator = torch.load(config["output"]["dis_chckpt_file"])
            generator = torch.load(config["output"]["gen_chckpt_file"])


    discriminator.to(device)
    generator.to(device)

    # Training information ##############

    n_epochs = config["training"]["num_epochs"]
    start_epochs = config["output"]["chckpt"] if load_chckpt else 0
    batch_size = config["training"]["batch_size"]
    n_evals = config["training"]["n_evals"]
    zn = config["training"]["latent_dim"]
    true_input_dim = config["training"]["input_data_len"]
    lr = config["optimizer"]["lr"]

    # optmizer definitions ##############

    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
    generator_optimizer = optim.Adam(generator.parameters(), lr=lr)

    # Binary Cross Entropy loss #########
    loss = nn.BCELoss()

    # metrics to keep track of ##########
    bin_acc = BinaryAccuracy()
    acc = Accuracy()

    Write_FullInfo(config, tensorboardSummary.writer)
    # Start of optimization loop ########
    for epoch_idx in range(start_epochs, n_epochs):

        # Reset metrics for every epoch.
        G_loss = []
        D_loss = []
        D_acc_true = []
        D_acc_gen = []

        # Metric that is very interesting
        D_loss_true = []
        D_loss_gen = []

        for batch_idx, data_input in enumerate(training_data):

            noise = torch.randn(batch_size, zn).to(device)
            generated_data = generator(noise) # batch_size X 76

            # Discriminator
            true_data = data_input[0].view(batch_size, true_input_dim).to(device) # batch_size X 76
            digit_labels = data_input[1] # batch_size
            true_labels = torch.ones(batch_size).to(device)

            discriminator_optimizer.zero_grad()

            discriminator_output_for_true_data = discriminator(true_data.float()).view(batch_size)
            true_discriminator_loss = loss(discriminator_output_for_true_data, true_labels)

            discriminator_output_for_generated_data = discriminator(generated_data.detach()).view(batch_size)
            generator_discriminator_loss = loss(
                discriminator_output_for_generated_data, torch.zeros(batch_size).to(device)
            )
            discriminator_loss = (
                true_discriminator_loss + generator_discriminator_loss
            ) / 2

            discriminator_loss.backward()
            discriminator_optimizer.step()

            D_loss.append(discriminator_loss.data.item())

            discriminator_acc_for_true_data = bin_acc(discriminator_output_for_true_data.cpu(), true_labels.cpu())
            D_acc_true.append(discriminator_acc_for_true_data)
            discriminator_acc_for_generated_data = bin_acc(discriminator_output_for_generated_data.cpu(), torch.zeros(batch_size).cpu())
            D_acc_gen.append(discriminator_acc_for_generated_data)


            D_loss_true.append(true_discriminator_loss.data.item())

            D_loss_gen.append(generator_discriminator_loss.data.item())

            # Generator ##################
            generator_optimizer.zero_grad()

            # It's a choice to generate the data again
            generated_data = generator(noise) # batch_size X 76
            discriminator_output_on_generated_data = discriminator(generated_data).view(batch_size)
            generator_loss = loss(discriminator_output_on_generated_data, true_labels)
            generator_loss.backward()
            generator_optimizer.step()

            G_loss.append(generator_loss.data.item())

            if (batch_idx + 1 == batch_size and (epoch_idx + 1)%n_evals == 0):

                tensorboardSummary.writer.add_scalar('Train/Loss/Discrim_loss', torch.mean(torch.FloatTensor(D_loss)), epoch_idx)
                tensorboardSummary.writer.add_scalar('Train/Loss/Gen_loss', torch.mean(torch.FloatTensor(G_loss)), epoch_idx)
                tensorboardSummary.writer.add_scalar('Train/Loss/Discrim_loss_true', torch.mean(torch.FloatTensor(D_loss_true)), epoch_idx)
                tensorboardSummary.writer.add_scalar('Train/Loss/Gen_loss_gen', torch.mean(torch.FloatTensor(D_loss_gen)), epoch_idx)
                tensorboardSummary.writer.add_scalar('Train/Accuracy/Discrim_true_acc', torch.mean(torch.FloatTensor(D_acc_true)), epoch_idx)
                tensorboardSummary.writer.add_scalar('Train/Accuracy/Discrim_gen_acc', torch.mean(torch.FloatTensor(D_acc_gen)), epoch_idx)

                with torch.no_grad():
                    noise = torch.randn(batch_size,zn).to(device)
                    generated_data = generator(noise).cpu().view(batch_size, true_input_dim)
                    tensorboardSummary.writer.add_histogram("Training/difference",
                                                            (generated_data - true_data.cpu()).flatten()*Integral,
                                                            global_step = epoch_idx)
                    tensorboardSummary.writer.flush()
                    generated_data = generated_data[0::4].reshape(5, 5, true_input_dim)
                    pdata = true_data[0::4].cpu().reshape(5, 5, true_input_dim)
                    fig = make_image_plot(generated_data, pdata, Integral, true_input_dim)
                    tensorboardSummary.writer.add_figure(config["plot_title"] + "/Training", fig[0], global_step = epoch_idx)
                    tensorboardSummary.writer.flush()

                torch.save(generator, os.path.join(out_path, "generator_model_epoch_" + str(epoch_idx) + ".pth"))
                torch.save(discriminator, os.path.join(out_path, "discriminator_model_epoch_" + str(epoch_idx) + ".pth"))
        print (f"Epoch[{epoch_idx} / {n_epochs}] \t D_loss : {torch.mean(torch.FloatTensor(D_loss))} \t G_loss : {torch.mean(torch.FloatTensor(G_loss))}")

    # add_graph() will trace the sample input through your model,
    # and render it as a graph.
    tensorboardSummary.writer.add_graph(generator, torch.randn(batch_size, zn).to(device))
    tensorboardSummary.writer.add_graph(discriminator, generator(torch.randn(batch_size, zn).to(device)).float())
    tensorboardSummary.writer.flush()

    TestModel(config, generator, discriminator, tensorboardSummary)





if __name__=='__main__':

    parser = argparse.ArgumentParser(description='JLab Interview Problem 2')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file...')

    args = parser.parse_args()

    config = json.load(open(args.config))

    TrainModel(config)
