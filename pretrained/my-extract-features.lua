require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'
require 'xlua'
local js = require 'cjson'
local t = require 'transforms'

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224),
}

-- M = {}
local FeatureExtractor = torch.class('FeatureExtractor') -- M)

-- get all the image files from the given image directory
local function get_list_of_filenames(img_dir)
  -- decide whether the directory exist or not
  if not paths.dirp(img_dir) then
     io.stderr:write('Image directory: ' .. img_dir ..  ' is not found.')
    os.exit(1)
  end

  local lfs  = require 'lfs'
  local list_of_filenames = {}

  for file in lfs.dir(img_dir) do
      -- get the list of the files
      if file~="." and file~=".." then
          table.insert(list_of_filenames, img_dir..'/'..file)
      end
  end
  return list_of_filenames
end


-- load model from the path
function load_model(model_path)
  -- Load the model
  local model = torch.load(model_path):cuda()

  -- Remove the fully connected layer
  assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
  model:remove(#model.modules)

  -- Evaluate mode
  model:evaluate()

  return model
end

-- Cnovert tensor to a json str
function get_json_str_of_tenson(matrix)

   rows = matrix:size(1)
   cols = matrix:size(2)

   res_str = {}
   for i = 1, rows do
      feat_str = ''
      for j = 1, cols do
         feat_str = feat_str .. matrix[i][j] .. ' '
      end
      table.insert(res_str, feat_str)
   end
    res_str = js.encode(res_str)
   return res_str
end


function FeatureExtractor:__init(model_path)
   self.model = load_model(model_path)
end

function FeatureExtractor:extract(img_dir, list_of_filenames, batch_size)
  -- get paths of images into list_of_filenames
  if img_dir ~= nil then
    list_of_filenames = get_list_of_filenames(img_dir)
  else
    if list_of_filenames == nil then
      io.stderr.write('No image file is found.')
      os.exit(1)
    end
  end

  local number_of_files = #list_of_filenames

  batch_size = batch_size or 1
  if batch_size > number_of_files then
    batch_size = number_of_files
  end

  model = self.model

  local features

  for i=1,number_of_files,batch_size do
     xlua.progress(i, number_of_files)
     -- batch numbers are the 3 channels and size of transform
     local img_batch = torch.FloatTensor(batch_size, 3, 224, 224)

     -- preprocess the images for the batch
     local image_count = 0
     for j=1,batch_size do
        img_name = list_of_filenames[i+j-1]

        if img_name  ~= nil and paths.filep(img_name) then
           image_count = image_count + 1
           local img = image.load(img_name, 3, 'float')
           img = transform(img)
           img_batch[{j, {}, {}, {} }] = img
        end
     end

     -- if this is last batch it may not be the same size, so check that
     if image_count ~= batch_size then
        img_batch = img_batch[{{1,image_count}, {}, {}, {} } ]
     end

     -- Get the output of the layer before the (removed) fully connected layer
     local output = model:forward(img_batch:cuda()):squeeze(1)


     -- this is necesary because the model outputs different dimension based on size of input
     if output:nDimension() == 1 then
        output = torch.reshape(output, 1, output:size(1))
     end

     if not features then
        features = torch.FloatTensor(number_of_files, output:size(2)):zero()
     end
     features[{ {i, i-1+image_count}, {}  } ]:copy(output)
  end

  return get_json_str_of_tenson(features)
end

-- return M.FeatureExtractor
return FeatureExtractor
