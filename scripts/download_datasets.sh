FILE=$1

if [[ $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" &&  $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "vangogh2photo" \
    && $FILE != "photo2art" && $FILE != "edges2shoes" && $FILE != "edges2handbags" && $FILE != "night2day" && $FILE != "facades" && $FILE != "birds" ]]; then
    echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, vangogh2photo, \
    photo2art, edge2shoes, edges2handbags, night2day, facades"
    exit 1
fi

if [[ $FILE == "photo2art" ]]; then
    mkdir -p ./datasets/$FILE
    Files=(monet2photo cezanne2photo vangogh2photo)
    Domain=(B C D)
    for ((i = 0; i <= 2; i++))
    do
        URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/${Files[$i]}.zip
        ZIP_FILE=./datasets/${Files[$i]}.zip
        wget -N $URL -O $ZIP_FILE
        mkdir -p ./datasets/${Files[$i]}
        unzip $ZIP_FILE -d ./datasets/
        if [[ ${Files[$i]} == "monet2photo" ]]; then
            mv ./datasets/monet2photo/trainB  ./datasets/photo2art/trainA
            mv ./datasets/monet2photo/testB  ./datasets/photo2art/testA
        fi
        mv ./datasets/${Files[$i]}/trainA  ./datasets/photo2art/train${Domain[$i]}
        mv ./datasets/${Files[$i]}/testA  ./datasets/photo2art/test${Domain[$i]}
        rm -rf ./datasets/${Files[$i]}
        rm $ZIP_FILE
    done

elif [[ $FILE == "edges2shoes" ||  $FILE == "edges2handbags" || $FILE == "night2day" || $FILE == "facades" ]]; then
    URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
    TAR_FILE=./datasets/$FILE.tar.gz
    wget -N $URL -O $TAR_FILE
    mkdir -p ./datasets/$FILE
    tar -zxvf $TAR_FILE -C ./datasets/
    echo "Start preprocessing dataset..."
    python ./split.py ./datasets/$FILE
    echo "Finished preprocessing dataset."
    if [ -d "./datasets/$FILE/testB" ];then
        rm -rf ./datasets/$FILE/train
        rm -rf ./datasets/$FILE/val
        rm $TAR_FILE
    fi
elif [[ $FILE == "birds" ]]; then
    fileid="1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ"
    filename="birds.zip"
    mkdir -p ./datasets
    cd ./datasets
    curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
    wget -N http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz -O CUB_200_2011.tgz

    echo Extracting Data...
    unzip birds.zip
    tar -xvf CUB_200_2011.tgz -C ./birds
    unzip ./birds/text.zip -d ./birds

    rm birds.zip
    rm cookie
    rm CUB_200_2011.tgz
    rm ./birds/text.zip
else
    URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/$FILE.zip
    ZIP_FILE=./datasets/$FILE.zip
    wget -N $URL -O $ZIP_FILE
    mkdir -p ./datasets/$FILE
    unzip $ZIP_FILE -d ./datasets/
    rm $ZIP_FILE
fi