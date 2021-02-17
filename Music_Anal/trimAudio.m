function [y1,fs] = trimAudio(filename, startpoint)
    info = audioinfo(filename);
    fs = info.SampleRate;
    range = [fs*startpoint,fs*(startpoint+60)];
    [y1,fs] = audioread(filename,range);
end
