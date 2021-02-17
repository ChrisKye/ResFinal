myDir = uigetdir; %gets directory
myFiles = dir(fullfile(myDir,'*.mp3')); %gets all wav files in struct
list = [79, 70, 21, 181, 6, 51, 26, 6, 6, 6, 151, 151, 6, 6, 6, 41, 211, 86, 46, 151, 16, 126, 181, 186, 211, 191, 31, 6, 181, 86, 211, 6, 126, 131, 41, 6, 6, 166, 251, 201];
errorcount = 0;
errorvals = zeros(1);
for k = 1:length(myFiles)
    baseName = strcat("Data/OG_Music/videoid_",string(k),".mp3");
    info = audioinfo(baseName);
    leng = info.Duration;
    if leng < (list(k)+60)
        errorcount = errorcount + 1;
        errorvals(errorcount) = k;
        list(k)+ 60 - leng
    end
end