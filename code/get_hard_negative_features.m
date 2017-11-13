% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_hard_negative_features(non_face_scn_path, feature_params, w, b)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);
n = feature_params.template_size/feature_params.hog_cell_size;
D = n^2*31;
threshold = 0.2;
scales = [1.0,0.8,0.6,0.5,0.25,0.125];
features_neg = zeros(5000,D);
index = 1;

for i = 1:num_images
	img = im2single(rgb2gray(imread(fullfile(non_face_scn_path,image_files(i).name))));
    if index > 5000
		break;
    end
    for s = 1:length(scales)
		scale_img = imresize(img,scales(s));
        hog_feat = vl_hog(scale_img,feature_params.hog_cell_size);
        for j = 1:size(hog_feat,1)-n
            for k = 1:size(hog_feat,2)-n
                temp_hog_feat = reshape(hog_feat(j:j+n-1,k:k+n-1,:),[1,D]);
                score = temp_hog_feat*w + b;
                if score > threshold && index <= 5000
                	features_neg(index,:) = temp_hog_feat;
                	index = index + 1;
                end
            end
        end
    end
end
features_neg = features_neg(1:index-1,:);
end