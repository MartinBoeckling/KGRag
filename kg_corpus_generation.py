'''
Title: RDF2Vec transformation script

Description:
This script transforms an edge dataframe containing the following columns into a vector
representation using the RDF2Vec algorithm:
    - from: ID of grid cell or ID of geometric object as starting node
    - to: ID of grid cell or ID of geometric object as target node
    - description: Description of edge relation between starting and target node
    - DATE: Date of geometric object (YEAR or monthly data)
    - ID: Associated ID of grid cell.
The script contains three main methods. The first method is the dataPreparation method
which splits the edge dataframe into a training datafram grouping the dataframe to the
column year and a transformation dataframe grouping to the columns year and id.
The second script is the kgTraining method in which the training dataframe is grouping
to the year 

Input:
    - dataPath: Path to folder with edge dataframes in format dir/.../dir
    - distance: Distance of node to other node
    - maxWalks: maximum number of walks per defined entity
    - train: Boolean value if RDF2Vec should be performed
    - chunk: 
    - save:

Output:
    - Transformer model embedding in format of pickle file
    - Vector representation of grid cell IDs in format of pickle file
'''
# import packages
import argparse
import pandas as pd
import numpy as np
from rdflib.util import guess_format
from rdflib import Graph as rdfGraph
import random
from tqdm import tqdm
from pathlib import Path
from igraph import Graph
from langchain.prompts import ChatPromptTemplate
from itertools import groupby
import multiprocessing as mp
from gen_ai_hub.proxy.langchain.init_models import init_llm
from constants import ai_core_client_id, ai_core_client_secret, ai_core_resource_group, ai_core_auth_url, ai_core_url
import os


def read_kg_file(file_path: Path) -> Graph:
        """Warpper method to read different KG files in. If the edge dataframe uses a rdflib supported file type,
        rdflib is used. In case the KG is provided in a DeltaTable format, those readers it is used to read the file in.

        Args:
            file_path (Path): File path for knowledge graph file

        Returns:
            Graph: Return tuple structure containing triple structure using the 
        """
        # check if variable dataPath is a directory or a file
        # Read a Delta Table containing the entities we want to classify
        rdf_file_format = guess_format(str(file_path))
        if rdf_file_format is not None:
            kg = rdfGraph()
            kg.parse(file_path)
            kg.close()
            # prepare knowledge graph
            edge_list = [triple for triple in kg]
            edge_df = pd.DataFrame(edge_list, columns=['subject', 'predicate', 'object'])
        else:
            if file_path.suffix == ".parquet":
                edge_df = pd.read_parquet(str(file_path))
            elif file_path.suffix == ".csv":
                edge_df = pd.read_csv(str(file_path))
            else:
                file_ending = file_path.suffix
                raise NotImplementedError(f"For provided file path {str(file_path)} the file format {file_ending} is not implemented for read")
        edge_df = edge_df[['subject', 'object', 'predicate']]
        edge_df = edge_df.to_records(index=False)
        kg_graph = Graph().TupleList(edges=edge_df, directed=True, edge_attrs=["predicate"])
        # return prepared edge dataframe
        return kg_graph

class kg_embedding:
    """
    A class for embedding knowledge graph entities using random walks or BFS strategies.

    This class provides functionality to train knowledge graph embeddings based on random walks or
    breadth-first search (BFS) paths over a knowledge graph. The embeddings are trained using the 
    Word2Vec algorithm on sequences of nodes and predicates extracted from the graph.

    Attributes:
        data_path (Path): Path to the input data files containing the knowledge graph.
        distance (int): Maximum distance (number of hops) for walk iterations.
        max_walks (int): Maximum number of walks for each entity.
        train (bool): Flag indicating whether to train the model.
        chunksize (int): Size of chunks for processing entities during training.
        save_path (Path): Directory where the results and trained model will be saved.
        cpu_count (int): Number of CPU cores to be used for multiprocessing.
        walk_strategy (str): Strategy for walks ('random' or 'bfs').
        graph (Graph): The knowledge graph structure loaded from the data.
        model (Word2Vec): The trained Word2Vec model.
        
    Methods:
        __init__(data_path, distance, max_walks, train, chunksize, save_path, cpu_count, walk_strategy):
            Initializes the kg_embedding class with the given parameters.

        predicate_generation(path_list):
            Generates a sequence of predicates for a given path from the knowledge graph.

        random_walk_iteration(id_number):
            Performs a random walk over the graph starting from a given node.

        bfs_walk_iteration(id_number):
            Performs a BFS-based walk iteration for a given node ID.

        kg_training(graph_data):
            Trains the knowledge graph embedding model on the provided graph data.
    """
    def __init__(self, data_path: str, distance: int, max_walks: str,
                 train: bool, chunksize: int, save_path: str, cpu_count: int,
                 walk_strategy):
        """
        Initializes the kg_embedding class with the given parameters.

        Args:
            data_path (str): Path to the data files containing the knowledge graph.
            distance (int): Maximum distance for walks in terms of graph hops.
            max_walks (int): Maximum number of walks to perform for each entity (-1 for unlimited).
            train (bool): If True, triggers training of the knowledge graph embeddings.
            chunksize (int): The size of the chunks for processing the entities in parallel.
            save_path (str): Directory path where the trained model and results will be saved.
            cpu_count (int): Number of CPUs to use for parallel processing.
            walk_strategy (str): The walking strategy for graph exploration ('random' or 'bfs').

        Raises:
            IndexError: If no parquet file is found in the data path.
        """
        # transform string to Path structure
        self.data_path = Path(data_path)
        # assign distance variable to class variable
        self.distance = distance
        # assign maximum walks to class variable
        self.max_walks = max_walks
        # assign train to class variable
        self.train = train
        # assign chunksize to class variable
        self.chunksize = chunksize
        # assign savepath to class variable
        self.save_path = Path(save_path)
        # assign cpu count to class variable
        self.cpu_count = cpu_count
        # assign walk strategy to class variable
        self.walk_strategy = walk_strategy
        # create logging directory Path name based on file name
        logging_directory = self.save_path
        # create logging directory
        logging_directory.mkdir(parents=True, exist_ok=True)
        # extract all file paths from directory
        # create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)
        graph_data = read_kg_file(self.data_path)
        corpus = self.kg_corpus(graph_data)
        self.kg_sentence(corpus)


    def predicate_generation(self, path_list: str) -> list:
        """
        Generates a sequence of predicates for a given path from the knowledge graph.

        Args:
            path_list (str): The list of edges (path) for which to generate predicates.

        Returns:
            list: A list of predicates and nodes in the form of a sequence.
        """
        # assign class graph to graph variable
        # assign class graph to graph variable
        graph = self.graph
        # extract predicate of edge given edge id stored in numpy
        pred_values = [e.attributes()['predicate'] for e in graph.es(path_list)]
        # extract node sequences that are part of the edge path and flatten numpy array
        node_sequence = np.array([graph.vs().select(e.tuple).get_attribute_values(
            'name') for e in graph.es(path_list)]).flatten()
        # delete consecutive character values in numpy array based from prior matrix
        node_sequence = np.array([key for key, _group in groupby(node_sequence)]).tolist()
        # combine predicate values and node sequences to one single array
        if node_sequence:
            path_sequence = []
            for index, value in enumerate(node_sequence):
                node_label = value
                edge_label = pred_values[index]
                path_sequence.append(node_label)
                path_sequence.append(edge_label)
                if index >= len(pred_values) -1:
                    last_value = node_sequence[-1]
                    path_sequence.append(last_value)
                    break
        else:
            path_sequence = []
        # return path sequence numpy array
        return path_sequence

    def random_walk_iteration(self, id_number: int) -> list:
        """
        Performs a random walk over the graph starting from the specified node ID.

        Args:
            id_number (int): The node ID from which to start the random walk.

        Returns:
            list: A list of walk sequences, each representing a series of nodes and predicates.
        """

        walk_iteration = 0
        walk_list = []
        while walk_iteration <= self.max_walks:
            walk_iteration += 1
            walk_edges = self.graph.random_walk(start=id_number, steps=self.distance, return_type="edges")
            path_sequence = self.predicate_generation(walk_edges)
            walk_list.append(path_sequence)
        
        return walk_list

    def bfs_walk_iteration(self, id_number: int) -> list:
        """
        Performs a breadth-first search (BFS) walk for the given node ID.

        Args:
            id_number (int): The node ID for which to perform the BFS walk.

        Returns:
            list: A list of walk sequences generated by the BFS algorithm.
        """
        # assign class graph variable to local graph variable
        graph = self.graph
        # assign class maxWalks variable to local maxWalks variable
        max_walks = self.max_walks
        # extract index of graph node
        node_index = graph.vs.find(id_number).index
        # perform breadth-first search algorithm
        bfs_list = graph.bfsiter(node_index, 'out', advanced=True)
        # iterate over breadth-first search iterator object to filter those paths out
        # defined distance variable
        distance_list = [
            node_path for node_path in bfs_list if node_path[1] <= self.distance]
        # create vertex list from distance list extracting vertex element
        vertex_list = [vertex_element[0] for vertex_element in distance_list]
        # check if all paths should be extracted
        if max_walks == -1:
            pass
        else:
            # limit maximum walks to maximum length of walkSequence length
            vertex_list_len = len(vertex_list)
            if vertex_list_len < max_walks:
                max_walks = vertex_list_len
            # random sample defined maximumWalk from vertexList list
            random.seed(15)
            vertex_list = random.sample(vertex_list, max_walks)
        # compute shortest path from focused node index to extracted vertex list outputting edge ID
        shortest_path_list = graph.get_shortest_paths(
            v=node_index, to=vertex_list, output='epath')
        # extract walk sequences with edge id to generate predicates
        walk_sequence = list(map(self.predicate_generation, shortest_path_list))
        # return walkSequence list
        return walk_sequence

    def kg_corpus(self, graph_data: Graph) -> None:
        """
        Constructs the knowledge graph corpus using the provided graph.

        Args:
            graph_data (Graph): The knowledge graph data as a graph structure.
        """
        # initialize Knowledge Graph
        self.graph = graph_data
        entities = [vertex.index for vertex in self.graph.vs]
        print(self.graph.summary())
        # initialize multiprocessing pool with cpu number
        pool = mp.Pool(self.cpu_count)
        # extract walk predicates using the walkIteration method
        if self.walk_strategy == "bfs":
            walkPredicateList = list(tqdm(pool.imap_unordered(self.bfs_walk_iteration, entities, chunksize=self.chunksize),
                                        desc='BFS Walk Extraction', total=len(entities), position=0, leave=True))
        elif self.walk_strategy == "random":
            walkPredicateList = list(tqdm(pool.imap_unordered(self.random_walk_iteration, entities, chunksize=self.chunksize),
                                        desc='Random Walk Extraction', total=len(entities), position=0, leave=True))
        else:
            raise NotImplementedError(f"No implementation of provided value for walk strategy: {self.walk_strategy}")
        # walkPredicateList = list(pool.imap_unordered(self.walkIteration, entities, chunksize=self.chunksize))
        # close multiprocessing pool
        pool.close()
        # build up corpus on extracted walks
        corpus = [
            walk for entity_walks in walkPredicateList for walk in entity_walks]
        return corpus

    def kg_sentence(self, corpus: list, model_name: str):
        llm_model = init_llm(model_name)
        prompt_template = ChatPromptTemplate.from_template(
            """Please provide me from an extracted triple set of a Knowledge Graph a sentence. The triple set consists of one extracted random walk.
            Therefore, a logical order of the shown triples is present. Please consider this fact when constructing the sentence.
            Please return only the constructed sentence from the following set of node and edge labels extracted from the Knowledge Graph: {triples}""")
        tasks = [prompt_template.format_prompt(triples=walk) for walk in corpus]
        with open("responses.txt", "w", encoding="utf-8") as file:
            for _, result in llm_model.abatch_as_completed(tasks):
                file.write(result + "\n\n")

if __name__ == '__main__':
    # initialize the command line argparser
    parser = argparse.ArgumentParser(description='RDF2Vec argument parameters')
    # add train argument parser
    parser.add_argument('-t', '--train', default=False, action='store_true',
                        help="use parameter if Word2Vec training should be performed")
    # add path argument parser
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='string value to data path')
    # add distance argument parser
    parser.add_argument('-d', '--distance', type=int, required=True,
                        help='walk distance from selected node')
    # add walk number argument parser
    parser.add_argument('-w', '--walknumber', type=int, required=True,
                        help='maximum walk number from selected node')
    # add chunksize argument
    parser.add_argument('-chunk', '--chunksize', type=int, required=True,
                        help="use parameter to determine chunksize for parallel processing")
    parser.add_argument('-save', '--savepath', type=str, required=True,
                        help="use parameter to save path for files")
    parser.add_argument('-cpu', '--cpu_count', type=int, required=True,
                        help="number of CPU cores that are assigned to multiprocessing")
    parser.add_argument('-walk', '--walk_strategy', type=str, required=True,
                        choices=["random", "bfs"],
                        help="number of CPU cores that are assigned to multiprocessing")
    # store parser arguments in args variable
    args = parser.parse_args()
    kg_embedding(data_path=args.path, distance=args.distance, max_walks=args.walknumber, train=args.train,
                chunksize=args.chunksize, save_path=args.savepath, cpu_count=args.cpu_count,
                walk_strategy=args.walk_strategy)