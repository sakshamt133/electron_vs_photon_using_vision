from torchvision import transforms

electron_path = 'D:\\Datasets\\GSoc\\Electrons\\SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5'
proton_path = "D:\\Datasets\\GSoc\\Photons\\SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5"

transform = transforms.Compose([
    transforms.ToTensor()
])