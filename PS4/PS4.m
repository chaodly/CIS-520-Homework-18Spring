clear
clc

%% Load Data

main = 'C:\Users\nizhe\Desktop\2018 Spring\CIS-520\PS4\ps4-kit\ps4-kit\Problem-4\';
initFileName = strcat(main, 'init.txt');
transFileName = strcat(main, 'trans.txt');
emitFileName = strcat(main, 'emit.txt');
testFileName = strcat(main, 'test_sents.txt');
posFileName = strcat(main, 'pos.txt');
vocFileName = strcat(main, 'vocabulary.txt');


initFileID = fopen(initFileName);
initFile = textscan(initFileID, '%f');
fclose(initFileID);
init = cell2mat(initFile);

transFileID = fopen(transFileName);
transFile = textscan(transFileID, '%f');
fclose(transFileID);
trans = reshape(cell2mat(transFile), 218, 218)';

emitFileID = fopen(emitFileName);
emitFile = textscan(emitFileID,'%f');
fclose(emitFileID);
emit = reshape(cell2mat(emitFile), 14394, 218)';

testFileID = fopen(testFileName);
testFile = textscan(testFileID,'%s', 'Delimiter', '\n');
fclose(emitFileID);
test_1 = string(testFile{1, 1}{1, 1});
test_2 = string(testFile{1, 1}{2, 1});
test_3 = string(testFile{1, 1}{3, 1});
test1 = split(test_1);
test2 = split(test_2);
test3 = split(test_3);

posFileID = fopen(posFileName);
posFile = textscan(posFileID, '%s');
fclose(posFileID);
pos = string(posFile{1});

vocFileID = fopen(vocFileName);
vocFile = textscan(vocFileID, '%s');
fclose(vocFileID);
voc = string(vocFile{1});

K = size(pos, 1);

%% Viterbi Algorithm

[z1, tag1, p1] = viterbi(K, init, trans, emit, voc, pos, test1);
[z2, tag2, p2] = viterbi(K, init, trans, emit, voc, pos, test2);
[z3, tag3, p3] = viterbi(K, init, trans, emit, voc, pos, test3);
