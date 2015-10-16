% Eyelash removal using bottom hat filter (Sandipan Banerjee)

%load image
filename = 'EyeImages/02463d1928.tiff';
name1 = strsplit(filename,'/');
image = im2double(imread(filename));
mediatedImage = medfilt2(image);

%get hairs using bottomhat filter
se = strel('disk',5);
hairs = imbothat(mediatedImage,se);
hairs1 = hairs > 0.07;

% remove noise
hairs2 = medfilt2(hairs1,[4 4]);

% label connected components
lab_mask = bwlabel(hairs2); 
stats = regionprops(lab_mask, 'MajorAxisLength', 'MinorAxisLength'); 

%identifies long, thin objects 
Aaxis = [stats.MajorAxisLength]; 
Iaxis = [stats.MinorAxisLength]; 
idx = find((Aaxis ./ Iaxis) > 1); % Selects regions that meet logic check 
out_mask = ismember(lab_mask, idx);
mask2 = imdilate(out_mask,ones(5,5));
I2 = roifill(image,mask2);

%save result
newF = 'EyelashRemoved';
if ~exist(newF,'dir')
    mkdir(newF);
end
strng = strcat(newF,'/',char(name1(2)));
imwrite(I2,strng);