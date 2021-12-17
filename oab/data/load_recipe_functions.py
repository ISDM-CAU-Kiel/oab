import os
from os import listdir
from os.path import isfile
from pathlib import Path
from oab.data.utils import _make_yaml,_append_to_yaml,_get_dataset_dict
from oab.data.load_dataset import _uncompress_files,_make_one_file,load_plain_dataset,load_dataset
from oab.data.load_image_dataset import _load_image_dataset
from oab.data.utils_image import mvtec_ad_suffixes, mvtec_ad_datasets, mvtec_ad_color_datasets, image_datasets, url_dict, reshape_dict
import yaml
from ruamel.yaml import YAML
from json import loads,dumps


def get_tabular_dataset_names():
 
    yaml=YAML(typ='rt')
    yaml_content = yaml.load(Path("./")/ Path(os.getcwd()).parent/"oab"/"data"/"datasets.yaml")
    return [(i,names) for i,names in enumerate(yaml_content)]
    

def load_own_tabular_dataset(dataset_name:str,dataset_format:str="csv",class_labels:str="last",
                       filenames_to_concatenate:list=None,
                       csv_header:int=0,urls_dataset:str=None,destination_filenames:list=None,dataset_folder:str=Path(os.getcwd()).parent/"notebooks"/"benchmark_tabular"/"datasets"):
  

 """Helper that adds a dataset name and information along with the pre-installed file used by oab.

  :param dataset_name:                Name of the dataset
  :param dataset_folder:              the folder in which dataset is stored
  :param class_labels:                Which is the "lables" column , default="last" 
  :param dataset_format:              type of dataset file , by default="csv"
  :param filenames_to_concatenate:    Mention if separate files e.g. ["training.csv","testing.csv"] 
  :param foldername:                  Name of your dataset folder
  :param csv_header:                  by default 0, specifies the first row to be column headers
  :param dataset_url:                 If user want's to download the dataset via url
  
  
  :returns: the name which is to be used when loading the dataset.
 """
 
 info =        {  
                   'name':dataset_name,
                    'foldername':dataset_name,
                    
                    'dataset_format':dataset_format,'filenames_to_concatenate':filenames_to_concatenate
                    ,'filename_in_folder':f"{dataset_name}.{dataset_format}"
                    , 'load_csv_arguments': {'header':csv_header}
                    ,'destination_filenames':destination_filenames
                    , 'class_labels':class_labels
                    ,'urls_dataset':urls_dataset
                   }
 
 yaml=YAML(typ='safe')
 yaml_content = yaml.load(Path(os.getcwd()).parent/"oab"/"data"/"datasets.yaml")
 yaml_content[dataset_name] = info
 yaml.dump(yaml_content,  Path(os.getcwd()).parent/"oab"/"data"/"datasets.yaml")
 
    
 _uncompress_files(info, dataset_folder)
 _make_one_file(info, dataset_folder)  
 cd = load_plain_dataset(info, dataset_folder)
    
 return cd
# The function above  stores dataset_information in datasets.yaml and returns a Classifcation object


def add_own_dataset(dataset_name: str, bw: bool =False):
  """
  Helper that adds a dataset name to some internal variables used by oab.
  This is a workaround as the functionality to load an own dataset where files 
  are stored in a directory was not yet available when the paper was submitted.

  :param dataset_name: Name of the dataset
  :param bw: If the dataset is black-and-white

  :returns: the name which is to be used when loading the dataset.
  """
  name_mvtec_prefix = "mvtec_ad_" + dataset_name
  if dataset_name not in image_datasets:
      image_datasets.append(name_mvtec_prefix)
      mvtec_ad_suffixes.append(dataset_name)
      mvtec_ad_datasets.append(name_mvtec_prefix)
      if bw:
        mvtec_ad_bw_datasets.append(name_mvtec_prefix)
        reshape_dict[name_mvtec_prefix]=(256,256)
      else:
        mvtec_ad_color_datasets.append(name_mvtec_prefix)
        reshape_dict[name_mvtec_prefix]=(256,256,3)
      # url is just used for downloading the dataset if not already available offline and  for reproducibility
      # url_dict['myImageDataset']= 'ftp://<myftpserverurl>/myImageDataset.tar.xz 

  return name_mvtec_prefix 
  
#above is a helper function for loading image dataset


    
def dataset_info_store(dataset_name,new_recipe, info_type :str,content:list=None):
    #print(content)
    if not Path(new_recipe).is_file():
        
        _make_yaml(new_recipe,dataset_name,['dataset'])
    yaml=YAML(typ='rt')                                       
    yaml_content = yaml.load(Path("./") /new_recipe) 
    if dataset_name not in list(loads(dumps(yaml_content)).keys()): 
        
        yaml_content[dataset_name]=[]
        yaml_content[dataset_name].append('dataset')
    
    if  info_type in ['standard_functions','custom_functions']:
             data=[]
             for i in content:
               function={'name':i[0],'parameters':i[1]} 
               if function not in data:
                 data.append(function) 
    if info_type=='anomaly_dataset':

        data={'arguments':{'normal_labels':content,'anomaly_labels':None}}
    if info_type=='sampling':

        data=content[0]   

    if info_type not in [str(*i)  for i in loads(dumps(yaml_content[dataset_name])) if isinstance(i,dict)]:
             yaml_content[dataset_name].append({info_type:data})   


                
    yaml.dump(yaml_content, Path("./") /new_recipe)
    
    return
    
#function above  is for storing standard preprocessing function,anomaly-dataset-conversion and sampling paramters in a new recipe file

def algo_params(algo,l):
    
 if algo["algo_class_name"] in ["CAEABOD","CAEKNN","CAELOF"]:
   for i,neighbors in algo["algo_parameters"].items():
          if i!="CAE_parameters":
                  
                  if list(neighbors.keys())==['n_neighbors']:
                    for n_nei,param in neighbors.items():
                         
                         for (index,values) in enumerate(param.items()):
                             if index==0:
                                 x=values[1]*l
                             if index==1:
                                parameter_value=int(max(values[1],x))
                                #sub=algo["algo_parameters"][i][n_nei]
                                #algo["algo_parameters"][i][n_nei]=parameter_value
                               
                                for (o,x) in enumerate(algo["algo_parameters"].items()):
                                    
                                    if o==0:
                                        return {x[0]:x[1],i:{n_nei:parameter_value}}
 else:
    return algo["algo_parameters"] 
    
    
def algo_params_tabular(algo,l):
 #print(algo["algo_class_name"])   
 if algo["algo_class_name"] in ["ABOD","KNN","LOF","AELOF"]:
   pass
   for i,j in enumerate(algo["algo_parameters"].items()):
        if i==2:
            pass
   for i,neighbors in algo["algo_parameters"].items():

              if i!="AE_parameters":
                    if list(neighbors.keys())==['n_neighbors']:
                        for n_nei,param in neighbors.items():
                            for (index,values) in enumerate(param.items()):
                                 if index==0:
                                     x=values[1]*l
                                 if index==1:
                                    parameter_value=int(max(values[1],x))
                                    for (o,x) in enumerate(algo["algo_parameters"].items()):
                                        if o==0:
                                            #print({x[0]:x[1],i:{n_nei:parameter_value},j[0]:j[1]})
                                            return {x[0]:x[1],i:{n_nei:parameter_value},j[0]:j[1]}

                    else: 
                        
                        for (index,values) in enumerate(neighbors.items()):    
                            #print(index,values)
                            if index==0:
                                     x=values[1]*l
                            if index==1:
                                    parameter_value=int(max(values[1],x))
                                    for (o,x) in enumerate(algo["algo_parameters"].items()):
                                        #print(x[0],o)
                                        if o==0:
                                            return {x[0]:parameter_value}
                                             
                                    
 else:
    return algo["algo_parameters"]  

# The two functions above help to update algorithm parameters dynamically(required only in supervised cases) during the algorithm run, for image and tabular algorithms 


def data_from_recipe(info_type,file):
    
    yaml=YAML(typ='rt')
    yaml_content = yaml.load(Path("./") / file)
    if info_type=='datasets':       # importing all information about datasets in a dictionary
        
        return_datasets={}
        for dataset_name in yaml_content: 
            
          if yaml_content[dataset_name][0] =='dataset':
                
                print(f"\n{dataset_name}------")
                return_datasets[dataset_name]=['data_object','sampling','anomaly_dataset','standard_funcs','custom_funcs']
                dataset_details={}
                for x in loads(dumps(yaml_content[dataset_name])):
                    if isinstance(x,dict):
                        for key,value in x.items():
                           dataset_details[key]=value
                           if key=='sampling':
                                 for typ,params in value.items():
                                    return_datasets[dataset_name][1]=[params,typ]
                           if key=='anomaly_dataset':
                                return_datasets[dataset_name][2]={'normal_labels':value['arguments']['normal_labels']}
                           if key=='standard_functions':
                                stand_func_list=[]
                                for function in value:
                                    stand_func_list.append([function['name'],function['parameters']])
                                return_datasets[dataset_name][3]=stand_func_list
                           if key=='custom_functions':
                                stand_func_list=[]
                                for function in value:
                                    cust_func_list.append([function['name'],function['parameters']])
                                return_datasets[dataset_name][4]=cust_func_list 
                #print(dataset_details)
                yaml=YAML(typ='rt')
                yaml.dump(dataset_details, Path("./") / "helper.yaml") 
                
                if str(file)[-15:-12] in ["ssi","usi"]:
                    if dataset_name in image_datasets:
                     cd=load_dataset(dataset_name=dataset_name,anomaly_dataset=False,preprocess_classification_dataset=False,dataset_folder=Path(os.getcwd()).parent/"notebooks"/"benchmark_image"/"datasets") 
                    else:
                     add_own_dataset(dataset_name)   
                     cd=_load_image_dataset(dataset_name=f"mvtec_ad_{dataset_name}" ,anomaly_dataset=False,preprocess_classification_dataset=False,dataset_folder=Path(os.getcwd()).parent/"notebooks"/"benchmark_image"/"datasets") 
                else:
                    if dataset_name in [i[1] for i in get_tabular_dataset_names()]:
                        yaml=YAML(typ='safe')
                        yaml_dict= yaml.load(Path("./") / Path(os.getcwd()).parent/"oab"/"data"/"datasets.yaml")
                        yaml_details= yaml_dict[dataset_name]
                        details={'dataset_name':dataset_name,'class_labels':yaml_details['class_labels'],
                         'dataset_format':yaml_details['dataset_format'],
                          'filenames_to_concatenate':yaml_details['filenames_to_concatenate'],
                           'csv_header':yaml_details['load_csv_arguments']['header'],
                            'urls_dataset':yaml_details['urls_dataset'],
                        'destination_filenames':yaml_details['destination_filenames'] }
                        #print(details)
                        cd= load_own_tabular_dataset(**details)
                    else:
                        raise Exception('Error! Please make sure your dataset information is contained in "datasets.yaml" !!!')
                
                #print(dataset_details)    
                if 'standard_functions' in dataset_details.keys() or 'custom_functions' in dataset_details.keys():
                    cd.perform_operations_from_yaml(filepath="helper.yaml")
                    print("standard/custom preprocessing performed!")
                if 'anomaly_dataset' in dataset_details.keys():
                  if str(file)[-15:-12] in ["ust","usi"]:
                    ad=cd.tranform_from_yaml(filepath="helper.yaml",unsupervised=True)
                  else:
                    ad=cd.tranform_from_yaml(filepath="helper.yaml",semisupervised=True)
                  print("transformed to anomaly dataset!")
                  return_datasets[dataset_name][0]=ad 
                
                os.unlink("helper.yaml") 
        return return_datasets
                

    if info_type=='algos':     # importing algo names and hyperparameters 
        algo_list={}                
        for i in yaml_content:
            
            if yaml_content[i][0]=='algo_name':
                x=yaml_content[i][-1]
                yaml_algo_content= loads(dumps(yaml_content[i][1]))
                yaml.dump(yaml_algo_content, Path("./") /f"{i}.yaml")
                algo_list[i]=[yaml_algo_content,f"{i}.yaml"]
                os.unlink(f"{i}.yaml")
        all_recipe_algos=[]
        for algo in algo_list:
          print(f"{algo}----\n")
          recipe_algo_details={}
          
          recipe_algo_details["algo_module_name"]=algo
          #print(algo)
          recipe_algo_details["algo_class_name"]=x
           
          recipe_algo_details["algo_name_in_result_table"]=x
          
          for i,j in enumerate(algo_list[algo][0].values()):
                if i==1:
                   recipe_algo_details["fit"]=j
                elif i==2:
                    
                    if str(file)[-15:-12] in ["ust","usi"]:
                   		recipe_algo_details["decision_scores"]=j
                    else:
                               recipe_algo_details["decision_function"]=j
                else:
                    for k in j.values():
                      recipe_algo_details["algo_parameters"]=k
          all_recipe_algos.append(recipe_algo_details)  
        return all_recipe_algos
    
    if info_type=='seed':           # importing seeds information for reproducibility
        seed=loads(dumps(yaml_content['seed']))[0]
        
        return seed


#function above is for obtaining dataset and algorithm information from recipe file



def sample_unsupervised(dataset_name,sampling_type,sampling_params,data_obj):    
    
    if sampling_type == 'unsupervised_multiple':
      return data_obj.sample_multiple(**sampling_params)
    elif sampling_type == 'unsupervised_single':
      return data_obj.sample(**sampling_params)  
    elif sampling_type == 'unsupervised_multiple_benchmark':
      return data_obj.sample_multiple_benchmark(**sampling_params)  
    else:
        raise NotImplementedError(f"Sampling from yaml with type {type} is not implemented.")      

def sample_semisupervised(dataset_name,sampling_type,sampling_params,data_obj):    
    
    if sampling_type == 'semisupervised_multiple':
      return data_obj.sample_multiple(**sampling_params)
    elif sampling_type == 'semisupervised_explicit_numbers_single':
      return data_obj.sample_with_explicit_numbers(**sampling_params)  
    elif sampling_type == 'semisupervised_training_split_multiple':
      return data_obj.sample_multiple_with_training_split(**sampling_params)  
    elif sampling_type == 'semisupervised_training_split_single':        
      return data_obj.sample_with_training_split(**sampling_params)   
    else:
        raise NotImplementedError(f"Sampling from yaml with type {type} is not implemented.")      
#The two functions are  for sampling depending on type chosen 
